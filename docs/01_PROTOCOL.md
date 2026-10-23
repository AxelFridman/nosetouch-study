# 01 Experimental protocol (stage 1)

This document applies to `PROTOCOL_VERSION` 1.0.0, app 0.1.0, schema 1.1.0 and consent `2026-10-v1`, as set in [constants.ts](../app/src/core/constants.ts).

**Sources of truth.** The code defines the protocol. This document explains it.

| What | File |
|---|---|
| Plans: presets, block order, randomisation | [protocol.ts](../app/src/core/protocol.ts) |
| Participant wording | [common.ts](../app/src/i18n/en/common.ts) (protocol text) and [session.ts](../app/src/i18n/en/session.ts) (screen text) |
| Trial and session logic | [trialController.ts](../app/src/core/trialController.ts), [segmentation.ts](../app/src/core/segmentation.ts), [sessionFlow.ts](../app/src/app/sessionFlow.ts) |
| Data contract | [records.schema.json](../schema/records.schema.json), explained in [02_DATA_SCHEMA.md](02_DATA_SCHEMA.md) |

Every session record stores its seed, full plan and parameters, so each session can be reproduced from its own record. If this document and the code disagree, the code is right and this document must be fixed.

**Conventions.** Statements about the app describe the code at the versions above. **Recommendation** marks a proposal that is not yet implemented or decided. Literature and platform claims link to their sources.

**Related documents:**
- [00_PROJECT_OVERVIEW.md](00_PROJECT_OVERVIEW.md): entry point
- [03_SENSORS_AND_PLATFORMS.md](03_SENSORS_AND_PLATFORMS.md): sign and axis conventions, sampling rates
- [04_SEGMENTATION.md](04_SEGMENTATION.md): the movement detector
- [05_MODELING_PLAN.md](05_MODELING_PLAN.md): mirroring, evaluation, power
- [06_CONSIDERATIONS.md](06_CONSIDERATIONS.md): ethics, safety and risk checklists
- [07_DEPLOYMENT.md](07_DEPLOYMENT.md): hosting and the upload endpoint
- [08_STAGE3_PREDICTOR.md](08_STAGE3_PREDICTOR.md): the stage-3 predictor

**Contents.** 1 Definitions · 2 Apparatus · 3 Factors and instructions · 4 Presets · 5 Order and randomisation · 6 Session flow · 7 Trial timeline · 8 Review and label quality · 9 Safety · 10 Participants and sample size · 11 Validation with phyphox · 12 Lab-run vs remote · 13 Ethics prerequisites · 14 Suggested improvements · 15 Open questions for the PI

---

## 1. Scope and definitions

**The PI's question.** A person holds the phone with the arm fully extended in front and brings it to touch their nose. Can the phone's motion sensors alone tell whether they used the left or the right hand?

**What stage 1 produces.** Stage 1 records this movement across many people, phones, phone orientations, speeds and deliberately odd movements. It produces two things:
- a training set for stage 2;
- an error map: under which conditions, and for which people, the left/right decision fails.

| Term | Definition (data field) |
|---|---|
| hand | The hand that holds the phone and performs the movement (`condition.hand`: `left`/`right`). It is set per block by the plan, and it is the target label. |
| handedness | A participant trait recorded in the profile: `handedness_self`, plus the EHI-SF quotient and class (`ehi_lq`, `ehi_class`). `hand` is never derived from it. |
| dominant hand | The hand that matches the handedness class. The right hand is the dominant hand for about 90% of people, which is why everyone performs both hands (§10). |
| start pose | The arm fully extended in front, the phone in the instructed orientation, held still. Its lateral position is the factor `start_position` (§3.3). |
| nose contact | The phone gently touches the tip of the nose and stays there until the double beep. Which part of the phone touches is not standardised (§15). The `chin` and `forehead` variants change the target. |
| trial | One planned movement of the session plan (`trial_index`, global, contiguous from 0). |
| attempt | One recording of a trial (`trial_id`; `attempt` = 1, 2, …). Every attempt is stored, including redos and failures. |
| accepted | `outcome.accepted = true`: the attempt completed (`status = complete`) and the participant confirmed it ("Yes, next"). A finished trial has exactly one accepted attempt. |
| block | A run of consecutive trials with one hand and one orientation (`block_index`; `kind` = `main`, `odd` or `free`). |
| session | One pass through one plan, in one posture (`session_id`). |
| participant | Identified by a pseudonymous `participant_id`: "P-" plus 6 characters, without 0/O/1/I. It is generated on the phone. |

**Unit of observation.** The unit is one attempt. It is the full raw recording from the Ready tap (t = 0) to 1000 ms after the end cue, so it contains:
- the arm raise and the hold in the start pose;
- the GO cue and the movement;
- the hold at the nose and the start of the return.

The phone never crops the recording. The online detector `online_v1` only drives the cues and QA. The analysis re-segments every recording offline with several methods ([04_SEGMENTATION.md](04_SEGMENTATION.md)).

## 2. Apparatus

| Item | Specification |
|---|---|
| Phone | The participant's own iPhone or Android phone, with a gyroscope. An own phone is preferred over a lab phone because it adds device diversity and matches the PI's own-phone test. Without a gyroscope, `rotationRate` is null ([Chromium](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc)). **Recommendation:** exclude phones without a gyroscope. |
| Browser | On iPhone, any browser works: they all use WebKit. On Android, use Chrome or Samsung Internet. Brave on Android blocks motion sensors ([Brave forum](https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620)). In-app browsers are detected (`device.in_app_browser`) and handle the motion permission inconsistently ([WebKit bug 213468](https://bugs.webkit.org/show_bug.cgi?id=213468)). |
| App | The NoseTouch PWA. Participants open a link or QR code over HTTPS; nothing needs installing. |
| Motion stream | `devicemotion` (`acceleration`, `accelerationIncludingGravity`, `rotationRate`) at about 60 Hz, the browser maximum. The cap comes from WebKit's 1/60 s timer ([WebCoreMotionManager.h](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.h)) and Chromium's 60 Hz pump ([device_sensor_event_pump.h](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_sensor_event_pump.h)). Chrome rounds values to 0.1 m/s² and 0.1 °/s; WebKit does not round ([platform_sensor_util.h](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). |
| Other streams | `deviceorientation` Euler angles, kept as metadata only because of gimbal lock at the upright start pose ([W3C](https://w3c.github.io/deviceorientation/)). A RelativeOrientationSensor quaternion on Chromium/Android only, capped at 60 Hz ([sensor_traits.h](https://github.com/chromium/chromium/blob/main/services/device/public/cpp/generic_sensor/sensor_traits.h)). Screen touches and cue times. |
| Cues | Synthesised beeps, large on-screen colour cues and text, and a screen wake lock. There is no vibration while recording. |

**Why 60 Hz is enough.** For minimum-jerk movements lasting at least 0.25 s, 99% of the acceleration energy lies below about 10 Hz (our calculation on the [minimum-jerk profile](https://www.socsci.ru.nl/wilberth/physics/smooth.html)). Native apps sample faster: about 100 Hz on iOS ([WWDC23 session 10179](https://developer.apple.com/videos/play/wwdc2023/10179)) and 200–500 Hz on Android ([Android sensors](https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview), [phyphox forum](https://phyphox.org/forums/printthread.php?tid=2356)). §11 tests whether that difference matters.

Platform details are in [03_SENSORS_AND_PLATFORMS.md](03_SENSORS_AND_PLATFORMS.md).

## 3. Factors and participant instructions

Quoted text is verbatim from [common.ts](../app/src/i18n/en/common.ts) or [session.ts](../app/src/i18n/en/session.ts). The Hebrew version in `app/src/i18n/he/` asks for proofreading in its file header. **Recommendation:** have a native speaker check it before recruiting Hebrew speakers, because instruction wording changes how people move. The language used is stored per session in `settings.language`.

### 3.1 Overview

| Factor | Levels | Varies per | Field |
|---|---|---|---|
| hand | `left`, `right` | block | `condition.hand` (target) |
| orientation | 10 `OrientationId` values (§3.6) | block | `condition.orientation` |
| speed | `natural`, `fast`, `slow` | trial | `condition.speed` |
| start position | `natural`, `shoulder`, `center` | trial | `condition.start_position` |
| variant | `standard` + 9 odd variants | trial (odd blocks) | `condition.variant` |
| posture | `standing`, `sitting` | session | `condition.posture`, `protocol.posture` |
| rep | 0, 1, …: repeats of an identical condition within a block | derived | `condition.rep` |

Models never get these fields as input ([predictor.ts](../app/src/core/predictor.ts)). `hand` is the target. The other factors define strata for the error analysis ([05_MODELING_PLAN.md](05_MODELING_PLAN.md)).

General instruction (`movementSummary`):

> Stretch your arm out and hold still. At the beep, bring the phone to gently touch the tip of your nose and hold it there until the double beep.

Before each trial, the screen shows that trial's phone orientation, speed, start position and movement, plus this hint (`ready.hint`):

> Tap Ready, stretch your arm out and hold still. Move only when you hear the beep.

### 3.2 Hand

- **Instruction at block start** (`holdWith`): "Hold the phone in your LEFT hand" or "Hold the phone in your RIGHT hand". When the hand changes, the block intro adds "Switch hands!".
- **Confirmation.** The label is the instructed hand. The participant confirms it on each attempt with "Yes, next", or contradicts it with the `wrong_hand` flag (§8).
- **Never inferred from handedness.** People hold the phone to the ear on their dominant side only about 70% of the time ([Seidman 2013](https://sciencedaily.com/releases/2013/05/130516161655.htm); survey figures from a press release, so low confidence).
- **Recommendation:** add to the briefing that the other hand does not touch the phone during a recording. The current wording implies this but does not say it.

### 3.3 Start position

| `start_position` | Label | Instruction |
|---|---|---|
| `natural` | Arm straight in front | "Stretch your arm straight out in front of you, at shoulder height." |
| `shoulder` | In line with your shoulder | "Arm fully straight, hand directly in front of the same shoulder." |
| `center` | In front of your face | "Arm fully straight, phone centered in front of your nose." |

Why the start position matters:

- **From `shoulder`**, the path includes a sideways shift of about half a shoulder width: leftward for the right hand, rightward for the left hand. Its sign flips with the hand, so it is an easy cue. Phone accelerometers recover it reliably only for short movements, though. In the digest's simulation, the sign of a 15 cm shift came out right in about 100% of runs up to 1 s, but only in 67–95% of runs at 3 s. That simulation is low confidence; see [Bai 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823748/) for background.
- **From `center`**, there is almost no sideways shift, so the hand must be recognised from rotation. Two findings suggest this is possible:
  - hand-to-nose movements involve about 54° of forearm rotation ([Öhberg 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6427602/));
  - IMU finger-to-nose studies report counter-clockwise rotation for the right hand and clockwise rotation for the left ([Mohammadi-Ghazi 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9609238/)). The rotation there is about a wrist-sensor axis that the paper does not state is vertical, so treat it as a hypothesis for the pilot to test.
- **`natural`** is the participant's own reading of the PI's literal instruction. It can fall anywhere between the other two and varies most between people. The simulator models it as a per-person blend (`start_blend`) of the other two.
- **Consequence:** report accuracy separately for `shoulder` and `center`. A model that works only from `shoulder` has mainly learned the sideways shift.

### 3.4 Speed

| `speed` | Instruction | Expected one-way duration | Detector overrides (`resolveParams`) |
|---|---|---|---|
| `natural` | "Move at your normal, comfortable pace." | About 0.5–0.9 s. The outward phase of finger-to-nose takes 0.62 ± 0.12 s in healthy adults ([Chen 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8072355/)). | none |
| `fast` | "Move quickly, as fast as you comfortably and safely can. Touch your nose gently!" | About 0.3–0.6 s. Fast 50 cm pointing takes 276–651 ms, depending on target difficulty ([Terrier 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3045308/)). | `onset_sustain_ms` 40 |
| `slow` | "Move slowly and smoothly, taking about 3 seconds." | About 3 s, as instructed. | `onset_gyro_min_dps` 15, `onset_acc_min_ms2` 0.3, `end_gyro_dps` 10, `end_acc_ms2` 0.25, `end_sustain_ms` 450, `max_movement_ms` 12000 |

The app instructs the speed but does not enforce it. The achieved duration is stored (`segmentation.duration_ms`) and shown in the review. **Recommendation:** analyse error by measured duration as well as by instruction.

**Why slow trials need lower thresholds.** The table shows expected peaks for a minimum-jerk movement over 0.45 m with 60° of rotation (our calculation):

| One-way duration | 0.35 s | 0.7 s | 3 s |
|---|---|---|---|
| Peak hand speed | 2.4 m/s | 1.2 m/s | 0.28 m/s |
| Peak linear acceleration | 21 m/s² | 5.3 m/s² | 0.29 m/s² |
| Peak angular speed | 321 °/s | 161 °/s | 38 °/s |

- **Slow trials.** At 3 s, the linear acceleration is close to Chrome's 0.1 m/s² rounding step and to postural tremor (about 0.14 m/s², [Elble 2005](https://smithengineering.queensu.ca/mme/faculty/deluzio/jam/files/Elble.pdf)). The gyroscope therefore carries slow trials.
- **Fast and odd trials.** These can exceed the ±4 g range of some phone accelerometers ([Galaxy A20 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9547012/)) and can reach 500–700 °/s. QA flags clipping ([02_DATA_SCHEMA.md](02_DATA_SCHEMA.md) §12).

### 3.5 Movement variants ("odd" movements)

**How odd blocks run:**
- There is one odd block per hand, after all main blocks.
- Every odd trial uses orientation `upright_toward`, start `natural` and speed `natural`, with one trial per variant.
- The block intro reads: "In this block every movement is a little different and unusual. Read each instruction before you tap Ready."
- Main blocks use the `standard` variant: "Bring the phone straight to your nose."

| `variant` | Instruction | Presets | Override | What it probes |
|---|---|---|---|---|
| `curved` | "Take a curved path: swing out to the side first, then to your nose." | standard, full | none | path shape |
| `pause` | "Stop halfway for about one second, then continue to your nose." | standard, full | `end_sustain_ms` 1800, `max_movement_ms` 12000 | two-part movement; end detection |
| `twist` | "Rotate (twist) the phone in your hand while bringing it to your nose." | standard, full | none | extra rotation in a direction the participant chooses (adversarial for rotation features) |
| `elbow_high` | "Raise your elbow high out to the side as you bring the phone in." | standard, full | none | different arm kinematics |
| `bent_start` | "Start with your elbow slightly bent instead of fully straight." | standard, full | none | a violated start pose |
| `lean_in` | "Lean your head forward to meet the phone halfway." | standard, full | none | shorter hand path; head motion |
| `chin` | "This time touch your CHIN instead of your nose." | standard, full | none | different target |
| `forehead` | "This time touch your FOREHEAD instead of your nose." | full only | none | different target |
| `jerky` | "Move to your nose in a jerky, stop-and-go way." | standard, full | `end_sustain_ms` 600 | multi-peak movement; stresses segmentation |

The hand label stays valid in every variant. Whether odd movements should also form a class of their own is open (§15). A pause longer than about 1.8 s counts as the end of the movement: the double beep sounds during the pause, so redo such attempts.

### 3.6 Phone orientation

The motion axes are fixed to the phone and do not follow screen rotation ([W3C](https://w3c.github.io/deviceorientation/)):
- x points to the right edge of the screen;
- y points to the top edge;
- z points out of the screen.

The same movement therefore projects onto different axes in each orientation. Rotations are described from the participant's point of view. "Toward" means the screen faces the participant, so the main rear camera faces away.

| `OrientationId` | Rotation | Screen | Mirror counterpart | Up axis at rest (`accelerationIncludingGravity`, W3C sign) | Presets |
|---|---|---|---|---|---|
| `upright_toward` | `upright` | `toward` | `upright_toward` | +y, reads (0, +9.8, 0) | Q S F |
| `upright_away` | `upright` | `away` | `upright_away` | +y | Q S F |
| `top_left_toward` | `top_left` | `toward` | `top_right_toward` | +x | Q S F |
| `top_right_toward` | `top_right` | `toward` | `top_left_toward` | −x | Q S F |
| `upside_down_toward` | `upside_down` | `toward` | `upside_down_toward` | −y | S F |
| `upside_down_away` | `upside_down` | `away` | `upside_down_away` | −y | F |
| `top_left_away` | `top_left` | `away` | `top_right_away` | −x | F |
| `top_right_away` | `top_right` | `away` | `top_left_away` | +x | F |
| `flat_up` | `flat` | `up` | `flat_up` | +z | F |
| `free` | participant's choice | any | `free` | any | F (free blocks) |

Presets: Q = `quick`, S = `standard`, F = `full`. `custom` can use any orientation.

**Reading the up-axis column.** At rest, the accelerometer reads the reaction to gravity. That reaction points up, so the column names the device axis that points up.

**iPhones.** WebKit, which every iOS browser uses, reports `acceleration` and `accelerationIncludingGravity` with all axes negated; for example, z ≈ −9.81 when the phone lies flat with the screen up. Sources: [WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [W3C list, 2014](https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html), and a re-measurement in 2026 ([gaitscope #96](https://github.com/soroushdty/gaitscope/issues/96)). Calibration measures this per phone and stores it as `gravity_sign` = −1 (§6). Raw values are stored as delivered, and normalisation happens offline ([03_SENSORS_AND_PLATFORMS.md](03_SENSORS_AND_PLATFORMS.md)).

| `OrientationId` | Label | Instruction |
|---|---|---|
| `upright_toward` | Upright, screen facing you | "Hold the phone upright (top edge up) with the screen facing you, as if reading it." |
| `upright_away` | Upright, screen facing away | "Hold the phone upright with the screen facing away from you (the main camera looks at you)." |
| `top_left_toward` | Sideways, top to the left, screen facing you | "Turn the phone sideways so its top edge points to your LEFT, screen facing you." |
| `top_right_toward` | Sideways, top to the right, screen facing you | "Turn the phone sideways so its top edge points to your RIGHT, screen facing you." |
| `upside_down_toward` | Upside down, screen facing you | "Hold the phone upside down (top edge pointing down), screen facing you." |
| `upside_down_away` | Upside down, screen facing away | "Hold the phone upside down with the screen facing away from you." |
| `top_left_away` | Sideways, top to the left, screen facing away | "Turn the phone sideways with its top edge to your LEFT and the screen facing away from you." |
| `top_right_away` | Sideways, top to the right, screen facing away | "Turn the phone sideways with its top edge to your RIGHT and the screen facing away from you." |
| `flat_up` | Flat, screen up | "Hold the phone flat like a tray, screen facing the ceiling." |
| `free` | Your choice | "Hold the phone however feels most natural to you." Free blocks open with: "In this block, hold the phone in whatever way feels natural to you." |

**Mirror counterpart.** Reflecting the body across its sagittal plane turns a right-hand attempt in orientation o into a left-hand attempt in `ORIENTATIONS[o].mirror`. In other words, `top_left` and `top_right` swap, and the other orientations map to themselves. For any grip, the device-frame readings transform as:
- acceleration: (x, y, z) → (−x, y, z);
- angular velocity: (x, y, z) → (x, −y, −z). In `rotationRate` terms, `alpha` is kept and `beta` and `gamma` are negated, because alpha, beta and gamma are the rates about device x, y and z ([W3C CRD 2025](https://www.w3.org/TR/orientation-event/), [PR #43](https://github.com/w3c/deviceorientation/pull/43)).

[05_MODELING_PLAN.md](05_MODELING_PLAN.md) §3 gives the derivation.

**Protocol rule.** Keep any custom orientation set closed under this map: with `top_left_toward`, include `top_right_toward`. That way, mirrored copies land on conditions that also exist as real data. All presets already satisfy this rule.

**How the orientation is checked.** After each attempt, the app classifies the phone's rotation from the mean `accelerationIncludingGravity` of the hold-still window. The dominant axis must lie within 35°, and the result is stored in `detected.rotation_start`. The app compares it with the instructed rotation and stores the outcome in `detected.orientation_match`. The check has limits:
- In portrait, gravity cannot tell "toward" from "away".
- In landscape, a facing error shows up as a `top_left`/`top_right` swap.
- `free` has no instructed rotation, so the match is null.
- If calibration was skipped or its flat step found no sign, the check uses the sign the browser engine is known to report: −1 for WebKit (every iOS browser outside the EU) and +1 for Blink and Gecko. The same applies when the flat sign contradicts the engine. Only an unknown engine falls back to the W3C sign, which would read an upright iPhone as upside down. Calibration step a is still recommended, because it measures the sign instead of assuming it (§6).

### 3.7 Posture

The participant picks `standing` ("Standing") or `sitting` ("Sitting") at setup. There is one posture per session, stored in every trial.

**Recommendations:**
- **Standing is the reference posture:** facing forward, the free arm hanging.
- **Sitting** is allowed when a participant prefers or needs it. They sit upright, pushed back from any table so the arm extends freely, with the elbow unsupported.
- **Keep posture independent of the recruitment channel.** If lab participants stand and remote participants sit, posture and setting become confounded. If you want both postures, assign them per participant, or add a second session in the other posture.

The final choice is the PI's (§15).

## 4. Presets and session length

| Preset | Main-block orientations | Trials per main block (`speed`, `start_position`) | Odd blocks | Free blocks | Blocks | Trials | Estimate |
|---|---|---|---|---|---|---|---|
| `quick` | `upright_toward`, `upright_away`, `top_left_toward`, `top_right_toward` | 3: (natural, natural), (fast, natural), (slow, natural) | none | none | 8 | 24 | 7.5 min |
| `standard` (default) | the 4 above + `upside_down_toward` (`CORE_ORIENTATIONS`) | 4: (natural, shoulder), (natural, center), (fast, natural), (slow, natural) | 1 per hand: 8 variants (all except `forehead`) | none | 12 | 56 | 14 min |
| `full` | all 9 except `free` | 6: {natural, fast, slow} × {shoulder, center} | 1 per hand: all 9 variants | 1 per hand: speeds natural/fast/slow, start natural | 22 | 132 | 29.5 min |
| `custom` | any set | `speeds` × `start_positions` × `reps` (1–10) | optional | optional | 2 per orientation, plus 2 odd and 2 free | varies | formula below |

Main and free trials use the `standard` variant. We verified the counts and estimates by running `buildPlan` with seed 42.

- **Estimate** (`estimateMinutes`). It is 60 s of setup, plus 9 s per trial, plus 25 s per block change, rounded to 0.5 min, and is stored as `protocol.estimated_minutes`. It leaves out first-time onboarding (consent, profile, sensor check, calibration) and redos; measure both in the pilot.
- **Default custom spec.** `upright_toward` and `upright_away` × three speeds, start `natural`, 1 rep: 12 trials in 4 blocks (4 min).
- **Spec normalisation.** Custom specs pass through `normalizeCustomSpec`:
  - unknown values are dropped and duplicates removed;
  - values are put in canonical order;
  - empty lists get the defaults;
  - `reps` is clamped to 1–10.

  The normalised spec is stored in `protocol.custom`.

**Design caveats** of the presets as built:
1. `standard` does not cross speed with start position. `fast` and `slow` always start `natural`, and `natural` speed always starts `shoulder` or `center`. Within `standard`, speed and start-position effects therefore cannot be separated. Use `full` or a crossed custom design when that matters.
2. `full` main blocks never use start `natural`. That start appears only in the odd and free blocks.
3. Every preset records one trial per (hand, orientation, speed, start) cell per session.
4. Odd variants exist only in `upright_toward`, at natural speed.
5. `upright_toward` always comes first, and odd and free blocks always come last. These conditions are therefore confounded with time in session. Use `block_index` or `trial_index` as a covariate.

## 5. Block order, counterbalancing and randomisation

Rules from [protocol.ts](../app/src/core/protocol.ts):

1. **Block.** A block is one hand × one orientation.
2. **Orientation order.** `upright_toward` comes first as the familiarisation pose. The other orientations follow in seeded random order.
3. **Hands.** Hands follow an ABBA pattern over orientation pairs: A·o1, B·o1, B·o2, A·o2, A·o3, B·o3, …, where A is the first hand. This roughly balances time in session between the hands, so fatigue, learning and drift do not track the label.
4. **Odd and free blocks.** Odd blocks follow all main blocks (A, then B); free blocks come last (A, then B).
5. **Trial order within a block.** Trials within a block are shuffled. `rep` counts repeats of an identical condition in presentation order.
6. **First hand.** Right if the seed is even, left if it is odd (`resolveFirstHand`). The code's `firstHand` option overrides this.
7. **Random stream.** There is one random stream: mulberry32 (`makeRng`) seeded with `mixSeed(seed, 0x504c414e)` ("PLAN"). The orientation order consumes it first, then each block's shuffle in block order. First hand and posture do not consume the stream, so overriding the first hand only swaps the hands within the same plan.
8. **Seed.** Each session gets a fresh cryptographic uint32 seed (`randomSeed`), unless `createSession` is given one. The seed is stored in `protocol.seed`, together with `protocol_version`, `first_hand` and the full `plan`.
9. **Foreperiod.** Each attempt's foreperiod comes from `makeRng(trialSeed(seed, trial_index, attempt))` ([sessionFlow.ts](../app/src/app/sessionFlow.ts)), so it is reproducible too.

**Example** (`standard`, seed 42, from `buildPlan`): R·`upright_toward` → L·`upright_toward` → L·`top_right_toward` → R·`top_right_toward` → R·`upright_away` → L·`upright_away` → L·`top_left_toward` → R·`top_left_toward` → R·`upside_down_toward` → L·`upside_down_toward` → R·odd → L·odd.

**Balance across participants.** Over 2,000 consecutive seeds:
- exactly 50% of sessions start with the left hand;
- all 24 orders of the four shuffled `standard` orientations occur, at similar rates.

Within an actual sample, random seeds balance the first hand only in expectation. With 40 participants, the number who start with the left hand has an SD of about 3.

**Recommendation:** in lab sessions, pass seeds of alternating parity for exact balance. `createSession` accepts a seed, but the setup screen does not expose one (`SetupScreen.tsx` calls `createSession({ preset, posture, custom })`), so this needs a small UI change first.

## 6. Session flow

| Step | Screen | What happens | Stored in |
|---|---|---|---|
| 1 | Welcome | Choose or create the participant on this phone; choose the language | (nothing) |
| 2 | Consent | Consent text (`consent_version`); 18+ only (`min_age` = 18) | `session.consent` |
| 3 | Profile | Participant code, demographics, EHI-SF | `ParticipantProfile`, also snapshotted into the session |
| 4 | Sensor check | Motion permission (on a tap), event check, rate measurement, engine and in-app-browser detection | `device`, `capabilities` |
| 5 | Calibration | Three short poses (table below) | `calibration` |
| 6 | Setup | Preset (default `config.default_preset` = `standard`), posture, custom spec | `protocol`: seed, plan, estimate |
| 7 | Session | Blocks: intro → trials → break screen | trial records |
| 8 | Done | Summary, upload status, "Save / share my data" | `status`, `ended_at` |

**Profile.** The profile records the following (full definitions in [02_DATA_SCHEMA.md](02_DATA_SCHEMA.md) §4.2):
- the participant code `P-XXXXXX`, which participants keep so they can request deletion;
- an optional nickname, kept on the phone unless `upload_nickname` is set;
- age, sex (`female`, `male`, `other`, `prefer_not_to_say`) and `handedness_self`;
- the EHI-SF;
- `phone_hand`, the hand that usually holds the phone;
- optionally height, an arm or hand condition (`arm_issue`), the phone model and whether the phone has a case.

**EHI-SF** ([Veale 2014](https://www.millisecond.com/library/v7/edinburghhandednessinventory/edinburghhandednessinventory_sf.manual), [validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8317198/)):
- **Items.** Writing, throwing, toothbrush and spoon.
- **Answers.** Always right = 100, usually right = 50, both equally = 0, usually left = −50, always left = −100.
- **Scoring.** LQ is the mean of the four answers. LQ ≥ 61 is classed right, ≤ −61 left, and anything else mixed.

Item answers are stored, so the inventory can be re-scored later.

**Sensor check:**
- **Permission.** The app requests the motion permission from a tap on every browser that has `requestPermission`: iOS, and Chrome from version 151 ([release notes](https://developer.chrome.com/release-notes/151)).
- **Explain before the prompt.** On iOS, a denial is kept in memory until Safari is fully closed (swiped away in the app switcher); deleting the site's website data is not a reliable reset ([WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp)). The screen therefore explains the prompt before showing it.
- **After a denial.** The recovery advice depends on where the app runs: a Safari tab (close Safari in the app switcher and open the link again), the Home Screen app (close it in the app switcher and open it again from the Home Screen), another iOS browser (close that browser completely) or an in-app browser (open the link in Safari). It never suggests a Private tab: that is separate, temporary storage, so the participant would get a new code and lose the session in progress.
- **No events.** The check records for 2.5 s. If no motion data arrive in that window it reports `no_events`; below 20 Hz the participant cannot continue, and below 45 Hz the screen warns (`MIN_MOTION_HZ`, `GOOD_MOTION_HZ` in `SensorCheckScreen.tsx`).
- **Engine.** WebKit is detected from `webkitCompassHeading`, never from `requestPermission`.
- **Measured.** The sample rate, the interval median and 95th percentile, and which motion fields are present.
- **Recommendation:** continue only if `rotation_available` is true and `motion_hz` ≥ 50.

**Calibration** ([calibration.ts](../app/src/core/calibration.ts)). The app measures each phone's conventions rather than assuming them.

| Step | Participant | Pass criterion | Result | Expected |
|---|---|---|---|---|
| a. Flat | Puts the phone flat on a table, screen up, hands off. | Still: per-axis SD < 0.25 m/s² for accelerationIncludingGravity and < 3 °/s for rotationRate, with at least 20 samples. Gravity magnitude between 8.6 and 11.0 m/s², with z carrying at least 90% of it. | `gravity_sign` (sign of z), `gravity_magnitude`, `gyro_bias_dps` | +1 on Android, −1 on iPhone |
| b. Upright | Holds the phone upright in front, screen toward them. | y × sign > 7 m/s² → `ok`; < −7 → `inverted`; otherwise `unclear`. | `upright_check` | `ok` |
| c. Tilt | Starts flat, tilts the phone sideways by at least 30°, then back. | The rotation axis seen in gravity matches the integrated `rotationRate`: cosine ≥ 0.85 and magnitude ratio 0.6–1.6. | `axis_check.verdict`, `units` | `w3c`, `deg_per_s` |

Step c does not depend on the accelerometer sign. `rotationRate` signs agree between iOS and Android ([gaitscope #96](https://github.com/soroushdty/gaitscope/issues/96)).

**Checks on the screen** (the screen numbers the steps 1–3):
- **Flat sign against the engine.** A flat result whose sign contradicts the browser engine (WebKit should give −1, Blink and Gecko +1; an unknown engine accepts either sign) is shown as a warning, not as success: the phone most likely lay screen-down. The screen asks the participant to lay it flat with the screen up and measure again. If they continue anyway, `calibration.notes` records `flat_sign_contradicts_engine`.
- **Upright `inverted`.** The flat and upright steps disagree, typically because one was done face-down or upside down. The screen asks the participant to repeat steps 1 and 2.

**Recommendation:**
- Repeat step a until `gravity_sign` is set.
- Flag for review any session whose verdict is not `w3c`/`deg_per_s` ([02_DATA_SCHEMA.md](02_DATA_SCHEMA.md) §12, rules S3–S4).

**During the session:**
- **Block start.** Each block opens with an intro that names the hand and orientation and adds "Switch hands!" when the hand changes. The `block_start` cue plays.
- **Between blocks.** A break screen says "Lower your arm, shake it out and rest for a moment." and names the next block.
- **Pause.** "Pause" offers "Pause, continue later" or "End the session now". Ending sets the status to `abandoned`. Either way, everything recorded so far is kept.
- **Back button.** The Android back button or back gesture stays inside the app. On most screens it goes where the screen's ← arrow goes, and only the welcome screen lets it leave the app. On the session screen it opens the Pause dialog; during a recording it first stops the trial, which is saved as a failed attempt and offered again.
- **Double taps.** The main button of consecutive screens sits in the same place, so taps within 450 ms of a screen change are ignored. A double tap on "Start" therefore cannot also press the next screen's "Ready" and start a recording before the trial's instructions were shown.
- **Resuming.** A paused session can be resumed from the start screen. By default, the store offers sessions updated in the last 24 h (`findResumableSession`). Progress is reconciled with the stored attempts on resume.

**Done.**
- **Summary.** The screen shows movements recorded, attempts and time, plus the upload status and "Save / share my data" (a JSON export).
- **Upload.** If an upload endpoint is configured ([07_DEPLOYMENT.md](07_DEPLOYMENT.md)), every record is queued in IndexedDB and sent as soon as possible. Otherwise the participant sends the exported file.
- **Local data are only a buffer.** Safari deletes a site's storage after 7 days without use ([WebKit](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)).

**Participant setup checklist.** **Recommendation:** show this before block 1.
1. Open the link in Safari (iPhone) or Chrome (Android), not inside another app.
2. Turn the volume up. On iPhone, beeps play even with the silent switch on, because the app sets the audio session to playback ([WebKit bug 237322](https://bugs.webkit.org/show_bug.cgi?id=237322)).
3. Turn Low Power Mode off. It caps iOS web-page frames at 30 fps, and its effect on motion events is undocumented ([WebKit bug 215745](https://bugs.webkit.org/show_bug.cgi?id=215745)).
4. Turn on the phone's rotation lock. The page cannot lock rotation itself on iOS ([MDN compat data](https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json)), and a screen re-layout during a landscape trial is avoidable.
5. Turn on Do Not Disturb. A call or an app switch hides the page, motion events stop ([W3C](https://w3c.github.io/deviceorientation/)), and the attempt ends as `interrupted`.
6. On iOS below 18.4, use the Safari tab, not the Home Screen icon: the wake lock does not work there ([WebKit bug 254545](https://bugs.webkit.org/show_bug.cgi?id=254545)).
7. Choose a clear space with a soft surface below (§9). Allow about 15–20 min for `standard`, including onboarding.

## 7. Trial timeline and cues

Defaults are `DEFAULT_SEGMENTATION_PARAMS`; §3.4–3.5 list the per-condition overrides. Notation in the table:
- |ω| is the smoothed gyroscope magnitude.
- |a| is the smoothed linear acceleration, high-passed from `accelerationIncludingGravity` when the browser gives no `acceleration`.
- μ and σ are the mean and SD over the hold-still window.

| # | Phase | Participant | App | Cue and message | Data |
|---|---|---|---|---|---|
| 1 | start (t = 0) | Taps Ready, then stretches the arm into the start pose. | Recording starts. | `ready` beep | `timing.t0_perf_ms`, `t0_epoch_ms` |
| 2 | `waiting_still` | Holds the start pose. | Waits until \|ω\| < 15 °/s and \|a\| < 0.35 m/s² hold continuously for 600 ms (`still_hold_ms`). | "Hold still" with a progress indicator; hint "Arm straight, keep still and wait for the beep" | `timing.still_ms` |
| 3 | `foreperiod` | Keeps still. | Waits a random 500–1200 ms (uniform, seeded). | "Hold still" | `timing.foreperiod_ms` |
| 4 | `armed` | Moves at the beep. | Detects onset when \|ω\| > max(30 °/s, μ + 6σ) or \|a\| > max(0.6 m/s², μ + 6σ) for 60 ms. The onset is back-dated to the first sample after the last still sample, but no earlier than 300 ms before GO. | `go` beep; "GO!" | `timing.go_ms`, `onset_ms` |
| 5 | `moving` | Brings the phone to the nose and holds it there. | Detects the end when \|ω\| < 20 °/s and \|a\| < 0.45 m/s² for 300 ms, at least 150 ms after onset. The end is back-dated to the start of that still run. | "Touch your nose" | `timing.end_ms` |
| 6 | `post_roll` | Holds until the double beep, then may lower the arm. | Records 1000 ms more after the end is confirmed. | `end` double beep; "Hold it there…", then "Done!" | `timing.end_detected_ms`, `stop_ms` |
| 7 | review (after `done`) | Checks the plot and decides (§8). | (none) | (none) | `outcome.*` |

- **Attempt length.** An attempt records roughly 4 s (fast) to 7 s (slow); this is our estimate. The session estimate allows 9 s per trial, including the review.
- **Still time at the nose.** At least 300 ms before the double beep. Counting the participant's reaction to the beep, the total is roughly 0.6–0.8 s.
- **Early movement.** Moving above the onset floors (30 °/s or 0.6 m/s²) before GO counts as a false start. The `error` tone plays, the screen says "Too early! Wait for the beep", `timing.false_starts` goes up by 1, and the hold restarts. Smaller movements restart the hold silently.
- **Touches.** Every screen touch is recorded in `streams.touches`. The first touchstart after GO is saved as `timing.first_touch_ms`: a candidate nose-contact time when the screen faces the participant.

| Cue | Sound | When |
|---|---|---|
| `ready` | 440 Hz, 80 ms | recording starts |
| `go` | 880 Hz, 160 ms | GO |
| `end` | two 660 Hz beeps of 90 ms, 70 ms apart | end of movement confirmed |
| `error` | 220 Hz, 250 ms | false start, no movement, too long |
| `block_start` | 523 Hz then 784 Hz, 100 ms each | new block (plus a vibration on Android) |

- **Cue log.** Every cue is logged in `streams.cues`. The `go` time equals `timing.go_ms`.
- **Vibration.** Vibration is used only at block start, because the motor contaminates the accelerometer. iOS has no Vibration API in any case ([WebKit bug 288846](https://bugs.webkit.org/show_bug.cgi?id=288846)).
- **Screen-away orientations.** The participant depends on the beeps alone, so the volume must be up.

| `status` | Cause | Screen text |
|---|---|---|
| `no_movement` | No onset within 6 s after GO (`max_wait_ms`) | "No movement was detected after the beep. Start moving as soon as you hear it." |
| `too_long` | No end within 8 s; 12 s for slow and pause (`max_movement_ms`) | "The movement did not finish in time. Bring the phone to your nose and keep it there until the double beep." |
| `aborted` | "Hold to stop" (press for about 1 s), or the Android back button or back gesture during the recording | "You stopped the trial." |
| `interrupted` | Page hidden (screen off, app switch), or sensors stalled | "The trial was interrupted: the screen turned off or the app went to the background." or "The motion sensors stopped sending data. If this keeps happening, close the app and open the link again." |

Failed attempts are stored with `accepted = false` ("This attempt is kept for quality control but does not count."). The same trial is then offered again.

## 8. Review, redo and label quality

**What the review screen shows.** After each completed attempt, the screen shows:
- "Recorded!";
- the movement time;
- a small plot of rotation speed and acceleration, with markers for the beep, the movement start and end, and touches.

It asks "Did you do it as instructed?" and offers "Yes, next", "Redo" or "Problem". "Problem" opens "What went wrong?" with three choices: "I used the wrong hand", "I held the phone the wrong way" and "Something else went wrong". If the detected rotation differs from the instructed one, the screen adds:

> The phone seemed to be held {detected} instead of {expected}. If you held it differently from the picture, tap “Problem” and choose “I held the phone the wrong way”.

| Event | `outcome.accepted` | `participant_flag` | Plan advances? |
|---|---|---|---|
| "Yes, next" on a complete attempt | true | null | yes |
| "Redo" | false | null | no: same trial, attempt + 1 |
| "Problem" → one of the three choices | false | `wrong_hand`, `wrong_orientation` or `other_problem` | no |
| A failed attempt (§7), then "Try again" | false | null | no |
| Review never answered (the app was hidden, closed or reloaded first) | false, with `note` = `review_interrupted` | null | no |

These rules come from `applyDecision`, `asFailedAttempt` and `asUnreviewed` in [sessionFlow.ts](../app/src/app/sessionFlow.ts). A complete attempt is saved as soon as its review opens, in the not-accepted `review_interrupted` form, and saved again with the participant's decision, which drops the note. Closing or reloading the app during the review therefore loses no recording. Every attempt is kept: `n_attempts` counts all of them and `n_accepted` only the accepted ones.

**Label quality comes from five layers:**
1. **Instruction.** Each block intro names the hand and the orientation.
2. **Participant confirmation.** Every attempt is reviewed, and the participant can raise a flag.
3. **Gravity.** The orientation check (`detected.orientation_match`, §3.6).
4. **Quality fields.** `quality.motion_hz`, `gaps_over_50ms`, `hidden_during`, `warnings`, and `timing.false_starts`.
5. **Offline checks.**
   - Trials are flagged when segmentation methods disagree by more than 100 ms, or when a movement has several peaks ([04_SEGMENTATION.md](04_SEGMENTATION.md)).
   - A model may flag a suspected wrong label. Such trials go to human review; they are never relabelled automatically, and never excluded on model evidence alone, because that would inflate accuracy.

**Deviation budget: 5–15%.** For comparison, in a supervised 7-day wrist-sensor study, 15.6% of participants wore a sensor on the wrong wrist at least once, on 6.9% of days ([PLOE, Sensors 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538999/)). A self-guided app will not do better.

The default exclusion rules and sensitivity variants are in [02_DATA_SCHEMA.md](02_DATA_SCHEMA.md) §12. For example, an orientation mismatch keeps the trial for hand models but removes it from per-orientation analyses. The full prevention and detection table is in [06_CONSIDERATIONS.md](06_CONSIDERATIONS.md) §4. Pre-register the rules before seeing any model result.

## 9. Safety

Instruction shown to participants (`safety`):

> Hold the phone firmly (doing this over a bed or sofa is a good idea) and always touch your nose gently.

**Recommendations.** The app already shows the quoted texts.
1. **Clear space.** Keep nothing within arm's reach. Stand or sit over or beside a bed or sofa, so a dropped phone lands softly. Never record while walking or near stairs.
2. **Firm grip.** Use a wrist strap if the case has one. Touch gently, especially in fast and jerky trials; the fast instruction itself says "Touch your nose gently!".
3. **Glasses.** A fast reach can hit the frame. Suggest taking glasses off, or approaching gently ([06_CONSIDERATIONS.md](06_CONSIDERATIONS.md) §3).
4. **Stopping.** Stop at any pain or discomfort. "Pause" is always available, and "Hold to stop" aborts a single trial. People with an arm or hand condition may skip the study. `arm_issue` is recorded, and these participants are analysed separately.
5. **Fatigue.** Holding the phone at arm's length is tiring, which is why the hold before GO lasts only about 1.1–1.8 s (600 ms of stillness plus a random 0.5–1.2 s foreperiod). The break screen asks participants to rest. In `full` (about 30 min), suggest a longer break after the main blocks.
6. **Hygiene.** If a lab phone is shared, wipe it between participants, since it touches the face.
7. **Consent.** The consent text names the minimal risks: bumping the nose and dropping the phone ([45 CFR 46.116 checklist](https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-46/subpart-A/section-46.116)).

## 10. Participants, recruitment and sample size

**Why both hands and deliberate recruitment of left-handers:**
- **Prevalence.** About 10.6% of people are left-handed: 9.3–18.1%, depending on the criterion ([Papadatou-Pastou 2020](https://research-repository.st-andrews.ac.uk/handle/10023/19889)).
- **Dominance confound.** Dominant and non-dominant reaches differ in path curvature and initial direction, and the difference is smaller in left-handers ([Sainburg 2002](https://link.springer.com/doi/10.1007/s00221-001-0913-8), [Przybyla 2012](https://pure.psu.edu/en/publications/dynamic-dominance-varies-with-handedness-reduced-interlimb-asymme/)). A model trained mostly on right-handers can learn dominance instead of side.
- **Random recruitment is not enough.** About 190 randomly recruited people would be needed to reach 20 left-handers.
- **Sex.** Men are somewhat more often left-handed (odds ratio 1.23, [Papadatou-Pastou 2008](https://wrap.warwick.ac.uk/29434/)). Balancing sex among left-handers therefore needs targeted recruitment of left-handed women.
- **Mixed-handers.** EHI class `mixed` forms its own group.

| Phase | N | Composition | Purpose |
|---|---|---|---|
| Self-test | the student and lab members | an iPhone and an Android phone | Debug the app; confirm the conventions on real devices (`gravity_sign` −1 on iPhone and +1 on Android; `axis_check` `w3c`); time the session |
| Pilot | 8–10 | both platforms; at least 2 left-handers if possible (**Recommendation**) | Estimate the error rate, the ICC of per-trial correctness, the deviation rate and the session time; test the instructions; run the mirror sign-pattern check ([05_MODELING_PLAN.md](05_MODELING_PLAN.md) §3) |
| Main | 40–60 or more | at least 20, ideally 30, left-handers by EHI-SF; iOS and Android each at least 30%; sex roughly balanced within each handedness group; 10–15% extra to cover exclusions (§8) | Training, subject-wise cross-validation, subgroup error analysis |
| Validation | 10–20, a subset of the main sample | lab-run, both platforms | Compare 60 Hz against native rates (§11) |
| Locked test set | about 20% of participants | assigned at enrolment by a rule fixed in advance; pilot participants excluded ([05_MODELING_PLAN.md](05_MODELING_PLAN.md) §6.4) | The final estimate, mimicking the PI's test on new people |

**Subgroup comparisons need people, not trials.** Trials from the same person are correlated, which the ICC (intraclass correlation) measures. With m analysed trials per person, the design effect is 1 + (m − 1)·ICC. Each person then counts as m / (1 + (m − 1)·ICC) independent trials.

How to read the table:
- **Sample size per person.** m = 40, the main-block trials of `standard`.
- **Test settings.** Two-sided α = 0.05 and power = 0.8.
- **Comparing two groups.** Detecting 5% vs 15% error needs about 141 independent trials per group.
- **Estimating accuracy.** Estimating 95% accuracy within ±3% needs about 203.

We recomputed these figures from the formulas in the handedness digest.

| ICC | Design effect | People per group to detect 5% vs 15% error | People to estimate 95% accuracy ± 3% |
|---|---|---|---|
| 0.05 | 2.95 | about 11 | about 15 |
| 0.10 | 4.9 | about 18 | about 25 |
| 0.20 | 8.8 | about 31 | about 45 |
| 0.30 | 12.7 | about 45 | about 65 |

- **People per subgroup.** Plan 20–30 or more people for every subgroup you want to compare, and take the ICC from the pilot. Power is then simulated for the GLMM; see [05_MODELING_PLAN.md](05_MODELING_PLAN.md) §7.
- **Handedness add-on.** The unit is the participant. With 15 vs 45 people, an AUC of 0.75 has a 95% CI of about ±0.16. With 30 vs 30, it is about ±0.12 ([Hanley & McNeil 1982](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747); our calculation).
- **Identity and device effects.** Personal movement style and phone artefacts inflate any evaluation that splits by record rather than by subject ([Saeb 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/), [Stisen 2015](https://portal.findresearcher.sdu.dk/da/publications/smart-devices-are-different-assessing-and-mitigating-mobile-sensi/)). Many phone models help. Every session of a person must carry the same `participant_id`.
- **Recruitment.** **Recommendation:**
  - Use the lab and department mailing lists and student groups.
  - Post flyers and posts with a QR code that explicitly ask for left-handers.
  - Screen for: 18 or older, a phone with a gyroscope, and no condition that makes the movement painful.
  - Keep a weekly table of handedness × platform × sex × age band ([06_CONSIDERATIONS.md](06_CONSIDERATIONS.md) §9).

## 11. Validation sub-study with phyphox

**Purpose.** The sub-study answers three questions:
1. Do the browser's 60 Hz, rounded data lose left/right information compared with native rates?
2. How much does onset and offset timing degrade at 60 Hz?
3. Do the axis and sign conventions agree with an independent app?

**Tool.** [phyphox](https://phyphox.org/download/) (RWTH Aachen) is free and open source (GPL-3.0). It runs on Android 5.0+ and iOS 12.0+. The current version is 1.2.1, from September 2026 ([release note](https://phyphox.org/?p=5037), [F-Droid](https://f-droid.org/en/packages/de.rwth_aachen.phyphox/)). It reaches about 100 Hz on iPhone and, depending on the device, up to about 500 Hz on Android ([phyphox forum](https://phyphox.org/forums/printthread.php?tid=2356)).

**Experiment file `nose-touch.phyphox`** (**Recommendation**; it is not in the repository yet):

| Element | Setting |
|---|---|
| Sensors | `accelerometer` (includes gravity), `linear_acceleration`, `gyroscope` (rad/s), `gravity`, `attitude` (quaternion). Set `rate="0"` and `rateStrategy="request"` on each, which passes raw OS samples through. The default `auto` switches to `generate`, which duplicates or drops points ([input docs](https://phyphox.org/docs/file-format/input/)). |
| Labels | Dropdown or edit fields that write numeric buffers, exported next to the sensor data ([user-input docs](https://github.com/phyphox/phyphox-docs/blob/main/docs/file-format/views/user-input.md)). Codes are below. |
| Run | A timed run with an acoustic countdown ([phyphox](https://phyphox.org/?p=2823)): about 1–2 s delay, then about 6 s of recording (8 s for slow). |
| Export and format | CSV export (a ZIP with one file per export set). Declare file format version 1.19: phyphox refuses files newer than it supports, and 1.20 would lock out phyphox 1.2.0 installs ([file format](https://phyphox.org/docs/file-format/)). |
| Distribution | The file is hosted next to the PWA and opened from an online QR code or a `phyphox://` link ([transferring experiments](https://phyphox.org/docs/transferring-experiments/)). The remote-access interface is never used outside the lab: it is plain HTTP without authentication ([remote interface](https://phyphox.org/docs/remote-interface/)). |

**Label codes.** The codes follow the app's canonical enum orders:

| Buffer | Codes |
|---|---|
| participant | a session number in the lab log, which maps it to `P-XXXXXX` (phyphox fields are numeric) |
| hand | 0 = left, 1 = right |
| orientation | 1 = `upright_toward`, 2 = `upright_away`, 3 = `top_left_toward`, 4 = `top_right_toward`, 5 = `upside_down_toward`, 6 = `upside_down_away`, 7 = `top_left_away`, 8 = `top_right_away`, 9 = `flat_up`, 10 = `free` |
| speed | 0 = natural, 1 = fast, 2 = slow |
| start | 0 = natural, 1 = shoulder, 2 = center |
| variant | 0 = standard, 1 = curved, 2 = pause, 3 = twist, 4 = elbow_high, 5 = bent_start, 6 = lean_in, 7 = chin, 8 = forehead, 9 = jerky |

**Procedure:**
1. **Cohort.** 10–20 lab-run participants on both platforms. Each uses the same phone, instructions and conditions in both apps.
2. **Calibration in phyphox.** Record the three calibration poses (§6) in phyphox too, so its own conventions are measured, not assumed.
3. **Instrument order.** Alternate instruments by block in an ABBA order, so fatigue does not favour either one.
4. **Movement in phyphox.** Movements are self-paced, because phyphox has no GO beep and no online segmentation. The participant holds still for about 1 s after recording starts, then moves, then holds at the nose until the run ends. Segmentation is done offline.
5. **Optional, Android only.** Try capturing with Chrome and phyphox in split-screen at the same time. Whether both get full-rate data this way is unverified, so check the recorded rates.

**Analysis:**
1. **Pre-specify** what counts as a material difference, for example a gap larger than the bootstrap CI of the difference.
2. **Native vs emulated rate.** Train and evaluate subject-wise on the phyphox data twice: once at the native rate, and once after emulating the PWA (60 Hz sample-and-hold, rounding to 0.1 m/s² and 0.1 °/s).
3. **Instrument comparison.** Compare the PWA recordings with the phyphox recordings of the same conditions.
4. **Timing.** Compare onset and offset estimates between the two rates.
5. **Decision rule.** Build a native app only if the difference is material.

## 12. Lab-run and remote sessions

| Aspect | Lab-run | Remote |
|---|---|---|
| Setting | experimenter present | participant alone, via link or QR code |
| Phone | the participant's own; a lab phone only as a fallback | own |
| Protocol fidelity | observed; deviations corrected and logged | participant flags and automatic checks only |
| Posture and space | controlled | variable; only the declared posture is known |
| Device diversity | low if a lab phone is used | high (helps leave-one-device-out) |
| phyphox validation | possible | not planned |
| Throughput | low | high |
| Use for | self-test, pilot, validation cohort, recruiting left-handers in person | the bulk of the main sample |

**Lab sessions (Recommendation):**
- Read the app's wording to participants; do not paraphrase it.
- Demonstrate the start positions and orientations, but not the movement's path or speed. Movement style is what the model must generalise over, and a demonstration imposes the experimenter's own style.
- Keep a lab log: date, `participant_id`, `session_id`, phone, posture, and each observed deviation with its `trial_index`.
- When you see an attempt go wrong, flag it with "Problem" → "Something else went wrong".
- Video for ground truth is not part of the app. It needs its own consent and ethics approval.

**Remote sessions (Recommendation):**
- Show the setup checklist (§6).
- Ask people to take part once and to keep their code. The code is stored per phone and per browser storage, so a second phone creates a new `participant_id`. On iOS, the Home Screen copy probably does too, since it likely has its own storage (an inference from WebKit's storage design; see the web-sensors digest). One person in two folds biases the evaluation.
- Before recruiting, configure the upload endpoint (`upload.mode` in `public/config.json`; see [07_DEPLOYMENT.md](07_DEPLOYMENT.md)). With `none`, data arrive only if participants send the export file themselves.

## 13. Ethics and privacy prerequisites

Full checklists are in [06_CONSIDERATIONS.md](06_CONSIDERATIONS.md) §1–2. The protocol-relevant points:

1. **Approval first.** Collect from anyone other than the student only after approval by the responsible Bar-Ilan faculty or department ethics committee ([Bar-Ilan research authority](https://research.biu.ac.il/he/node/22446)).
2. **Consent text.** It must cover purpose, procedure, duration, minimal risks, voluntary participation and withdrawal, the data collected, pseudonymisation, storage, retention and a contact person. When the text changes, update `consent_version` in `public/config.json`.
3. **Safeguards built into the app:**
   - an 18+ gate;
   - `P-XXXXXX` codes instead of names;
   - the nickname stays on the phone unless `upload_nickname` is set;
   - no camera, microphone or location;
   - a private Drive folder;
   - deletion on request, by participant code.
4. **Israeli privacy law.** Amendment 13 of the Protection of Privacy Law has applied since 14 August 2025 ([IAPP](https://iapp.org/news/a/israel-marks-a-new-era-in-privacy-law-amendment-13-ushers-in-sweeping-reform), [Library of Congress](https://www.loc.gov/item/global-legal-monitor/2025-11-17/israel-amendment-to-privacy-protection-law-goes-into-effect/)).
5. **Location of participants.** Recruit in Israel only, unless GDPR basics are added. Monitoring people in the EU triggers Art. 3(2)(b) of the [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj).

## 14. Suggested improvements to consider

All items are recommendations, ordered roughly by value for cost.

1. **Free-choice first trial** (for the handedness add-on). Before any hand instruction, ask the participant to "Touch your nose with the phone the way you normally would". Then ask which hand they used. This gives a cheap behavioural prior to combine with `phone_hand` and the EHI-SF. It needs a minor schema change, because this hand is reported rather than instructed.
2. **Explicit hand confirmation per block.** Make the block intro button read "I'm holding it in my LEFT hand" instead of "Start". It costs one tap per block and protects against whole blocks recorded with the wrong hand (§8).
3. **A balanced incomplete block design (BIBD) across participants.** Each participant does 3 of the 9 orientations, drawn from 12 sets in which every pair of orientations appears together exactly once (a BIBD with v = 9, b = 12, r = 4, k = 3, λ = 1). With 3 speeds × 3 reps, that is 54 trials, about 11 min by `estimateMinutes`, and the whole sample still covers every cell. Implement it as a custom spec per participant. In the lab, the experimenter picks the set. For remote sessions the app needs a small change, for example deriving the set from the seed.
4. **At least 3 repetitions per cell.** Today every cell is recorded once per session. Setting `reps` = 3 in a custom spec lets you estimate within-person variability, and per-condition error rates then rest on more than one trial per person. For example, the 5 core orientations × 3 speeds × 3 reps make 90 trials, about 18.5 min.
5. **A practice trial before block 1,** stored and marked as practice. This needs a block kind or a flag in the schema. At present, the learning curve falls on `upright_toward`, which always comes first.
6. **Check the orientation before GO.** Gravity is currently checked only after the attempt. Checking it during the hold would prevent wasted attempts.
7. **A longer hold at the nose.** If offline gravity estimation needs more rest data after the movement, delay the double beep by about 0.5 s, or raise `end_sustain_ms`. The segmentation digest recommends holding about 1 s at the nose.
8. **Ask which ear the participant uses for calls.** It is another cheap handedness prior ([Seidman 2013](https://sciencedaily.com/releases/2013/05/130516161655.htm)).
9. **Pre-register before opening the locked test set,** for example on OSF: the hypotheses, primary metric, exclusion rules and subgroup analyses.

## 15. Open questions for the PI

1. **Start pose.** Should the hand start in front of the same shoulder, or at the midline, and at what height? This sets the size of the sideways cue (§3.3). The app currently records all three readings.
2. **Contact point.** Which part of the phone should touch the nose in each orientation: the screen centre, the top edge, or the back near the camera? This defines the end pose, and whether contact registers as a screen touch.
3. **Evaluation format:**
   - Will you test with your own phone, and is it an iPhone or an Android phone?
   - Single trials, or several per test?
   - Which orientations and speeds, and will odd movements be included?
   - Is a one-minute calibration (flat, upright, tilt) allowed before predicting?
   - What accuracy counts as success?
4. **Posture.** Should the reference posture be standing or sitting, and do we need both?
5. **Ethics.** Which committee is responsible? Does the admission-stage pilot need formal approval? May the data be reused for publication?
6. **Odd movements.** Should they form a separate "invalid movement" class, or serve only as robustness tests?
7. **Scale.** Is 40–60 participants, with at least 20 left-handers, realistic within the admission timeline? If not, the subgroup claims must shrink accordingly.
