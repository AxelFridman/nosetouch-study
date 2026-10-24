# Stage 3: the predictor for the PI

This document covers what the PI will use to evaluate the work, how the Stage-2 model becomes an in-browser predictor, how to test it fairly, and the hand-off checklist. The decisions behind it (D*n*) are in [09 §5](09_STAGE2_3_PLAN.md#5-stage-3-the-predictor-link); the edits that plan proposed for this document (09 §11.2) are applied here.

Related: [Modeling plan](05_MODELING_PLAN.md) · [Stage 2 and 3 plan](09_STAGE2_3_PLAN.md) · [Protocol](01_PROTOCOL.md) · [Segmentation](04_SEGMENTATION.md) · interface [predictor.ts](../app/src/core/predictor.ts).

**Legend.** **Rec.** = recommendation; **Hyp.** = hypothesis. **Status:** Stage 1 implements only the `Predictor` interface. It ships no model on purpose, and everything else in this document is to be built in Stage 3 (milestones M3, M6 and M7 of [09 §8](09_STAGE2_3_PLAN.md#8-milestones-and-acceptance-criteria)).

---

## 1. What the PI uses

The PI uses the same PWA, from one link or QR code, with no install. It runs in a **predict mode**:
- **The link** is `?predict=1&model=<id>`, with `?mode=predict` as an alias, always at the canonical URL <https://axelfridman.github.io/nosetouch-study/?predict=1>, never a tunnel. Every QR code (report, PI sheet, deck) is generated from `[predictor].url` in `report/report.toml`, and the evaluation QR pins the model with `&model=`. `&sim=1` gives a silent desk demo (a known-answer check only), `&lang=he` the Hebrew interface (D22).

The steps:

1. **Open the link** in Safari (iPhone) or Chrome (Android). An in-app browser, an insecure page or a browser without motion sensors is blocked with "open in Safari/Chrome" and a copy-link button.
2. **One tap** enables the sensors (the motion permission on iOS; Chrome 151+ also exposes `requestPermission()`), the audio and the wake lock.
3. **The device check: 2.5 s upright** (D23). The PI holds the phone upright, screen facing them, for 2.5 s. It requires motion at 50 Hz or more, a gyroscope and a still window. The sign of the raw `accg_y` (with |`accg_y`| ≥ 7 m/s²) must equal the browser engine's (WebKit −1, Blink and Gecko +1). On a disagreement the check is repeated once with a picture of the pose; a second disagreement refuses (`sign_conflict`), because a wrong sign would invert every answer. An unknown engine needs two consistent checks and flags every result. The rotation-rate mapping is assumed to be W3C and verified on every attempt by the gravity closure (gate G3). The check provides no gyro bias (D9). There is no table step.
4. **Trial.** The PI holds the phone at arm's length in any grip. The cues match the collector: hold still, GO beep, bring the phone to the nose, hold it there for about 2 s until the double beep.
5. **Result** (D27):
   - the hand in large type with the probability of that hand, max(p, 1 − p), e.g. "RIGHT 91 %", and P(left) and P(right) in small type ("<1 %" and ">99 %" at the extremes). Colours: LEFT #d1495b, RIGHT #00798c, as in every figure;
   - a confidence label: "high" at 0.90 or more, "medium" from τ to 0.90, "not sure" below τ (still naming the more likely hand);
   - one line from the model file's reliability table, e.g. "On new people, answers between 90 and 97 % were right 118 of 124 times";
   - the evidence: the first-half yaw, the duration and the gravity closure;
   - a footer with the model id, the first 8 characters of its sha256, the app version and the pipeline versions;
   - a retry request with its reason when the recording or the movement cannot be used (§4).
6. **Scoring.** The prediction is saved with a timestamp before the "It was LEFT / RIGHT" buttons unlock. The local log exports CSV with the columns `ts, model_id, model_sha8, app_version, platform, browser, phone_model, person_code, slot, true_hand, predicted_hand, p_right, confidence, status, reasons, latency_ms, condition_note` ([09 §5.7](09_STAGE2_3_PLAN.md#57-network-and-privacy)).

The app never asks for the hand, grip, speed or variant: the model is blind to `condition` by construction (§2.4). Not in v1: a bootstrap range for p, aggregation over several attempts, a hands-free series. Every attempt stands alone, and nothing is learnt during a session.

## 2. Architecture

### 2.1 Components

```
PredictApp (app/src/predict/, loaded with a dynamic import only for ?predict=1)
 └─ TrialController (unchanged; the cue parameters of §2.4; a placeholder condition)
     └─ TrialRecord (streams; its online_v1 indices only drove the cues)
         └─ predictor.predict(trial)   bound to this page load's upright check + device info
              1 gate G1, recording: rate, gaps, hidden page, gyroscope ........ retry
              2 normalisation: the sign from the check, W3C axes, units
              3 drop repeated samples, linear resampling to 60 Hz
              4 gyro bias: mean of the 400 ms before GO when its norm < 3 deg/s (D9)
              5 gravity_v1: gyro propagation from the rest windows
              6 segmentation: offline_v1, identical to training ................ gate G2: retry
              7 features: feat_d1, over the approach window only (D11)
              8 model: bundled JSON (logistic, yaw_rule or hgb trees)
              9 calibration: temperature (Platt for unconstrained models)
             10 gates G3 (unusual movement) and G4 (low confidence) ............. PredictionV2
```

Code layout (D22, D24):
- `main.tsx` branches before `boot()` and loads `./predict/mount` with a dynamic `import()`. Predict mode therefore makes no `config.json` fetch, never opens the collector's IndexedDB, never starts the uploader, and does not grow the collector bundle.
- `app/src/predict/` holds `PredictApp.tsx`, `predictor.ts`, `log.ts`, `pipeline/` and `models/`. The Python side is `motionlab/deploy.py` and `scripts/export_model.py`.
- Models are bundled into the predict chunk: `app/src/predict/models/<id>.json` plus `registry.ts`, which maps each id to its sha256. There is no runtime `models/index.json`. The previous model stays bundled for rollback.
- Storage: a separate IndexedDB, `nosetouch-predict`, with `study_code: 'nose-touch-predict'` and a throwaway P-code.

### 2.2 Interface

The current contract in `app/src/core/predictor.ts`:

```ts
export interface Prediction { hand: Hand; p_right: number; model_id: string; }
export interface Predictor { readonly id: string; predict(trial: TrialRecord): Prediction; }
```

The proposed Stage-3 extension (**Rec.**) is additive; the existing types stay unchanged:

```ts
export interface PredictionV2 extends Prediction {
  p_left: number;
  status: 'ok' | 'abstain' | 'retry' | 'error';
  confidence: 'high' | 'medium' | 'not_sure';
  reasons: string[]; // reason codes of §4, e.g. 'rate_below_50hz', 'no_segment', 'low_confidence'
  reliability: { p_lo: number; p_hi: number; n: number; k: number } | null;
  evidence: { yaw_first_half_deg: number | null; duration_ms: number | null; closure_deg: number | null };
  model_sha256: string;
  latency_ms: number;
  versions: { norm: string; gravity: string; segmentation: string; features: string; model_format: string };
}
export interface AsyncPredictor {
  readonly id: string;
  predict(trial: TrialRecord): Promise<PredictionV2>;
}
```

- **Why async.** The integrity check uses `crypto.subtle`, which returns Promises, and an ONNX runtime would too ([inference-session.ts](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/inference-session.ts)). JSON models can stay synchronous behind the same interface.
- **Abstentions.** When the predictor abstains, `hand` still carries the arg-max, so a forced-choice score can always be computed.
- **Context.** Normalisation needs the gravity sign and the device's engine, which do not live in `TrialRecord`. Bind them when the predictor is created: `createPredictor(model, { check, device })` takes the result of the upright check, not `SessionRecord.calibration` (predict mode has no session calibration).

### 2.3 Model packaging

**Decision: JSON and TypeScript.** ONNX comes into play only under D14: a research rung that beats the default by more than the CI width and passes the TypeScript parity test.

| Option | Models | Artefact | Runtime cost | Notes |
|---|---|---|---|---|
| JSON weights | `YawSignRule`, logistic regression (including the antisymmetric variant) | about 1–5 KB | none (a dot product) | The default. |
| JSON trees | `HistGradientBoostingClassifier` | tens to hundreds of KB (**Hyp.**: depends on trees × depth) | none (tree walks) | Read from scikit-learn's private `_predictors`, so the scikit-learn version is pinned and a round-trip test runs (D24). |
| ONNX + onnxruntime-web | GBM or logistic regression via skl2onnx ([skl2onnx 1.20](https://onnx.ai/sklearn-onnx/supported.html)); CNN or GRU via a PyTorch export | model ≤ 2 MB | The CPU binary `ort-wasm-simd-threaded.wasm` is 14.2 MB in onnxruntime-web 1.30.0 ([package files](https://app.unpkg.com/onnxruntime-web@1.30.0/files/dist)). GitHub Pages cannot send the COOP/COEP headers that threads need, so it would run single-threaded. | Only under D14. |
| Hand-written TS forward pass | small 1D-CNN or GRU (≲ 50k parameters) | JSON weights, ≲ 200 KB | none | Avoids the 14 MB runtime, but needs a parity test against PyTorch. |

**Decision rule.** Ship the simplest model whose development accuracy lies inside the 95 % CI of the best one (paired person bootstrap); freeze it before the locked test (D15).

**The model file** (D26). The values are placeholders, not results. Its own sha256 lives in `registry.ts`, because a file cannot hold its own hash. It is written minified, with sorted keys, no NaN and no trailing newline, and `.gitattributes` marks it `-text` so a CRLF checkout keeps the hash.

```json
{
  "format": "nosetouch-model/1",
  "model_id": "lr-chiral-d1-v1",
  "kind": "logistic",
  "antisymmetric": true,
  "versions": {"norm": "norm_v1", "gravity": "gravity_v1", "segmentation": "offline_v1",
               "features": "feat_d1", "model_format": "1"},
  "resample_hz": 60,
  "quantise_ios": false,
  "feature_names": ["chi_yaw_first_half_deg", "chi_yaw_second_half_deg", "chi_yaw_rate_b1"],
  "scale": [1.0, 1.0, 1.0],
  "coef": [0.0, 0.0, 0.0],
  "intercept": 0.0,
  "calibration": {"method": "temperature", "temperature": 1.0},
  "abstain": {"tau": 0.65, "target_accuracy": 0.95},
  "ood": {"features": ["duration_ms", "peak_gyro_dps", "peak_acc_ms2", "total_rotation_deg",
                       "inv_tilt_change_deg", "qc_closure_deg"],
          "lo": [0, 0, 0, 0, 0, 0], "hi": [0, 0, 0, 0, 0, 0], "max_outside": 1, "closure_max_deg": 5.0},
  "reliability": [{"p_lo": 0.5, "p_hi": 0.6, "n": 0, "k": 0}],
  "training": {"data_snapshot": "sha256:...", "git_commit": "...", "n_people": 0, "n_trials": 0,
               "platforms": {"ios": 0, "android": 0},
               "metrics": {"dev_cv": {"accuracy": [null, null, null], "auc": [null, null, null],
                                      "ece": null, "brier": null},
                           "locked_test": {"accuracy": [null, null, null]}},
               "created_at": "..."},
  "intended_use": "One nose-touch movement by an adult; phone held in any fixed grip.",
  "limitations": ["Screen-away grips only synthesised", "Trained mostly on standing participants"]
}
```

### 2.4 Training and deployment parity

1. **Deployment rate.** Train the deployed model on 60 Hz PWA data. High-rate data, if a phyphox cohort is ever run, serve only for validation unless converted with the PWA emulation ([05 §2.2](05_MODELING_PLAN.md#22-resampling-rec)).
2. **Same normalisation.** The deploy pipeline is implemented identically in TypeScript and Python, so an iPhone's input looks like the training data. That includes the iOS quantisation when the model was trained with it (`quantise_ios`; off by default until ablation A3 decides, [05 §2.1](05_MODELING_PLAN.md#21-normalisation-norm_v2)), and the pre-GO gyro bias of D9 in both.
3. **Same segmentation: `offline_v1` on the recorded window, in both languages** (D10). Python is the reference; the TypeScript port uses fixed filter coefficients and scipy's `filtfilt` padding, and golden fixtures pin it. online_v1 cannot do this job: with the old predict-mode parameters it re-segments 0 of 18 pilot trials, and the protocol-2.0.0 natural parameters cut slow movements short ([09 §2.2](09_STAGE2_3_PLAN.md#22-the-predict-mode-parameters-of-08-24-cannot-re-segment-training-data)). If an exact port proves impractical, a deploy variant with moving-average filters (`offline_d1`) replaces `offline_v1` in both languages, and ablation A5 reports the cost.
4. **Same features: `feat_d1`** (D11). Norm → drop repeated samples → linear resampling to 60 Hz → pre-GO bias → `gravity_v1` → `offline_v1` bounds → features over the approach window only: the chiral yaw family (first half, second half, 5 time bins, signed peak), ω·â, helicity, the normalised triple-acceleration feature, and odd × even products with log duration and peak |ω|. Mirror-even descriptors feed only the gates. No zero-phase filtering, no ZUPT velocity, no Euler angles, compass or absolute heading.
5. **Golden-file parity** (D25). Python writes the golden file (`scripts/export_model.py --golden` → `app/tests/fixtures/predict_golden_<id>.json.gz`) with about 70 trials: real trials; synthetic edge cases (iOS sign, legacy axis mapping, rad/s, missing acceleration, gaps); 10 mirrored copies; 6 trials that must abstain. For each trial it stores the segmentation indices, per-channel checksums after resampling, the features, the logit, p, the status and the reasons, so a failure shows which stage diverged. Vitest tolerances: features and p within 1e-6 relative; segmentation indices identical, with ±1 sample allowed on at most 1 % of trials; status, reasons and hand identical.
6. **Integrity and versions.** At start-up the browser recomputes the model's sha256 with `crypto.subtle` and refuses on a mismatch (`model_integrity`) or when the model's versions differ from the pipeline's (`version_mismatch`). An unknown `&model=` id is refused.
7. **Condition blindness.** A test randomises `trial.condition`, and another gives it a `condition` getter that throws; the prediction must not change.

**Predict-mode cue parameters** (**Rec.**; validate on the rehearsal). online_v1 in predict mode only decides when to play the GO and end cues and when to stop recording; the model never sees its indices (D10).

| Parameter | Value |
|---|---|
| base | `DEFAULT_SEGMENTATION_PARAMS` of protocol 2.0.0 |
| `onset_gyro_min_dps`, `onset_acc_min_ms2` | 25, 0.55 (the slow values, so slow starts are caught) |
| `end_gyro_dps`, `end_acc_ms2` | 30, 0.6 (the natural values) |
| `end_sustain_ms` | 1800 (pauses do not end the recording early) |
| `max_movement_ms` | 12000 |
| `post_roll_ms` | 300 |
| `return_cap_ms` | 0 (predict mode does not record the return) |

- **Trade-off.** The person holds the phone at the nose for about 2 s, which also guarantees that the recording contains the whole movement and the hold.
- **Placeholder condition.** `TrialController` still needs a `TrialCondition`. Predict mode passes a fixed placeholder that the predictor never reads (item 7).
- **Keep predict-mode records separate:** the study code `nose-touch-predict`, a separate database, and no upload (§5).

### 2.5 Budgets (Rec.)

| Item | Budget | Check |
|---|---|---|
| Growth of the collector bundle | 0 KB: predict code sits behind a dynamic import | `vite build` output; a test that the collector entry chunk is unchanged |
| JSON model | ≤ 200 KB, bundled into the predict chunk | file size test |
| ONNX model (only under D14) | ≤ 2 MB, plus the lazy-loaded 14.2 MB runtime | network panel |
| Inference time | ≤ 50 ms for JSON models on a mid-range phone | `latency_ms` in every prediction |
| From the end of the trial to the shown result | ≤ 1 s (a 1-s watchdog on the computing state) | manual test |
| Offline | works after the first predict-mode visit; no network requests at all: before the sensors start, `mount()` inserts `<meta http-equiv="Content-Security-Policy" content="connect-src 'none'">` (after the predict chunk has loaded), which blocks fetch, XHR, WebSocket and `sendBeacon` for the life of the page (D28) | airplane-mode test; Playwright on Chromium and WebKit; a real iPhone |
| Hosting | far below the GitHub Pages limit of 1 GB per site ([limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)) | none |

## 3. An evaluation protocol the PI can run

### 3.1 Design (D29)

- **Attempts.** 12 per person, 6 with each hand. The hand order is on paper, in balanced blocks of 4 (2 left and 2 right, shuffled); the app never sees it.
- **Conditions.** The PI chooses them. The suggested list for each hand: upright, sideways, fast, slow, tilted touch, and one of the PI's own.
- **Scoring.** Each slot counts its first attempt that produced a prediction; retries are allowed only for gates G1 and G2. The truth is entered after the prediction has been saved.
- **Phone and people.** The PI's own phone; beforehand, state which development results apply to it (leave-one-platform-out and leave-one-device-model-out, [05 §6](05_MODELING_PLAN.md#6-evaluation-protocol)). More people with 12 attempts each give tighter intervals than one person with many.
- **Blinding.** The app does not know the hand, and the student neither chooses nor sees the hands during testing.
- **Rehearsal** (S3-M3). Before the PI, 3–5 people who were never in the training data run the same protocol. Their sessions form the stress set, and they include the conditions the collector never collects: screen facing away, sitting, walking, a phone case, landscape and upside down.
- **Stress tests (optional, unlabelled).** Shake the phone, lift it to the ear, put it down on the table. The desired behaviour is a retry or an abstention.

### 3.2 How to report

- **Primary metric:** forced-choice accuracy with an exact (Clopper–Pearson) 95 % CI.
- **Secondary metrics:** accuracy on answered attempts and coverage; accuracy with abstentions counted as errors (the conservative number); Brier score (a coin scores 0.25) and log loss (a coin scores 0.693), shown next to the development reliability diagram and ECE; accuracy per person. Conditions are reported descriptively.
- **What 12 attempts can show.** One person can show "better than chance", not a precise accuracy: 10 of 12 correct are needed for p < 0.05 (one-sided, against chance).

  | Correct | 95 % CI | One-sided p vs chance |
  |---|---|---|
  | 12/12 | 73.5–100 % | 0.0002 |
  | 11/12 | 61.5–99.8 % | 0.003 |
  | 10/12 | 51.6–97.9 % | 0.019 |
  | 9/12 | 42.8–94.5 % | 0.073 |

- **Several people.** With 5 or more people, use a person-level bootstrap. Otherwise give the pooled count with an exact interval, and state that it ignores the correlation between attempts of one person; five people at 57/60 give 86.1–99.0 %. With about 100 test trials, error bars of about ±10 % are normal ([Varoquaux 2018](https://arxiv.org/abs/1706.07581)).

Exact 95 % CIs for a pooled count (my computation):

| Correct / trials | 95 % CI |
|---|---|
| 19 / 20 (95 %) | 75–100 % |
| 38 / 40 (95 %) | 83–99 % |
| 95 / 100 (95 %) | 89–98 % |
| 90 / 100 (90 %) | 82–95 % |
| 180 / 200 (90 %) | 85–94 % |

Trials of one person are correlated, so the effective n is n / (1 + (m − 1)·ICC), with m trials per person. Half-width of a 95 % CI at 90 % accuracy (my computation):

| Plan | ICC 0.05 | ICC 0.1 | ICC 0.2 |
|---|---|---|---|
| 5 people × 20 trials | ±8 pts | ±10 pts | ±13 pts |
| 10 people × 10 trials | ±7 pts | ±8 pts | ±10 pts |
| 10 people × 20 trials | ±6 pts | ±7 pts | ±9 pts |

Report template:

| Field | Value |
|---|---|
| `model_id`, model sha256 (8 characters), `app_version` | |
| Phone: platform, model, browser | |
| People / attempts / answered | |
| Forced-choice accuracy (exact 95 % CI) | |
| Accuracy on answered attempts, coverage | |
| Accuracy with abstentions as errors | |
| Brier score, log loss | |
| Standard vs odd movements | |
| Per-person accuracy (min–max) | |
| Expectation committed beforehand | |

### 3.3 State the expectation in advance

Before the PI's session, the student generates the expectation as table T15 (expected forced-choice accuracy with its CI per platform and for standard vs odd movements, τ and the expected coverage) and commits it to `report/frozen/expectation_<model_id>.json` with the model's sha256. The commit time is the timestamp. The session then tests a stated prediction instead of producing a number to explain afterwards.

## 4. Failure handling

Abstaining beats a confident wrong answer. Every refusal, retry and abstention names its reason ([09 §5.6](09_STAGE2_3_PLAN.md#56-gates-and-failure-handling-changes-to-08-4)).

| Situation | Detection | Behaviour (reason code) |
|---|---|---|
| In-app browser, insecure page, no motion API | `device.in_app_browser`, capabilities | Block: "open in Safari/Chrome", with a copy-link button |
| Permission denied, no events | `requestMotionPermission`; no event within 2.5 s | Block, with the `deniedContext` advice (`permission_denied`, `no_events`) |
| The check fails | rate < 50 Hz, not still, \|`accg_y`\| < 7 m/s² | Repeat the check (`check_failed`) |
| No gyroscope | `rotation_available` false during the check | Refuse (`no_gyro`) |
| Check and engine disagree twice | D23 | Refuse (`sign_conflict`) |
| Model hash or versions wrong | D24 | Refuse (`model_integrity`, `version_mismatch`) |
| G1 recording | rate, a gap in [onset − 0.3 s, end + 0.3 s], hidden page, gyroscope | Retry, not scored (`rate_below_50hz`, `gap`, `hidden`, `no_gyro`) |
| G2 segment | no onset, too long, no `offline_v1` segment, duration outside 0.25–8 s or peak \|ω\| below 30 deg/s | Retry, not scored (`no_movement`, `too_long`, `no_segment`) |
| G3 unusual movement | the 0.5–99.5 % training ranges of 6 mirror-even descriptors (duration, peak \|ω\|, peak \|a\|, total rotation, tilt change, gravity closure): 2 or more outside, or the closure above its 99.5th percentile (at least 5°) | Abstain, with the more likely hand shown small (`unusual_movement`, `closure`) |
| G4 low confidence | max(p, 1 − p) < τ | "Not sure" (`low_confidence`) |
| An unconstrained model fails the mirror check | \|p(x) + p(m(x)) − 1\| > 0.2 | Confidence label at most "medium" |

- **The check replaces the calibration rows.** A missing or unclear sign is no longer met with "use the engine prior, low confidence": the check must agree with the engine, or the predictor refuses.
- **The OOD rule uses mirror-even descriptors only,** so abstaining cannot depend on the hand (D20). A Mahalanobis rule is ablation A12.
- **Removed:** "multi-peak → medium confidence" (v1 has no portable peak counter; the OOD rule covers odd movements).
- **Choosing τ.** τ is the smallest threshold at which the development out-of-fold accuracy on answered attempts reaches the target, e.g. 95 % (question b of [09 §10](09_STAGE2_3_PLAN.md#10-open-questions-for-the-pi)). Accuracy at τ on those same predictions is optimistic and labelled so; the honest numbers come from the locked test and the PI session. Never pick τ on the locked test.
- **Forced choice.** If the PI wants forced choice, set τ = 0.5 and keep the confidence label.

## 5. Privacy in predict mode

- **No uploads in predict mode** in v1 (D28): the CSP of §2.5 blocks every request, and the predict chunk does not even import the uploader. No profile is needed; an ephemeral P-code satisfies the schema.
- **Local log.** It keeps each raw `TrialRecord`, the prediction and the truth on the phone, and exports CSV and JSON through `core/exporter.ts`. Export is manual.
- **Central logging** (`&log=1`) is deferred: it needs consent text, and it would have to choose its CSP before the first one is inserted.
- **Testing other people.** Rehearsal attempts kept for the stress set are recorded data: tell those people what the local log keeps and get their agreement.

## 6. Hand-off checklist

1. The locked-test report is final, and the expectation per platform is committed (§3.3).
2. The model lives in `app/src/predict/models/` with its entry in `registry.ts`, and carries its versions, the data snapshot hash, the git commit and its metrics with CIs.
3. The golden-file parity test (TypeScript vs Python) passes.
4. The condition-blindness test passes.
5. Refusal on a version or hash mismatch is tested.
6. Predict mode works on iOS Safari and Chrome on Android, on at least 2 phones each and on the PI's model if known, including the 2.5-s upright check on both engines (both gravity signs) and Rotation Lock on iOS, since iOS cannot lock the screen orientation ([MDN BCD](https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json)).
7. The offline test passes: airplane mode after the first load.
8. Each reason code has been triggered once in a test, including no movement, too long, low rate, low confidence, `sign_conflict`, `unusual_movement` and `model_integrity`.
9. The latency and size budgets (§2.5) are measured and recorded.
10. Predict mode makes no network requests after loading, verified in the browser's network panel and with Playwright on Chromium and WebKit.
11. The one-page PI sheet, `report/pi_sheet.tex` (M6), covers the QR code, the steps, the scoring table, what counts as an error, and the known limitations.
12. Known limitations are listed, e.g. screen-away grips only synthesised, odd movements, slow movements from a `center` start, phones without a gyroscope, in-app browsers, browsers below 50 Hz.
13. The previous model is still bundled and can be selected with `&model=` (rollback).
14. `model_id`, the model's short sha256 and `app_version` are visible on the result screen.
15. A desk demo with `?predict=1&sim=1` works. It is silent, and it is a known-answer check only, not evidence of accuracy.

## 7. Open questions for the PI

1. Which phone or phones will you test with (platform, model)? We need to make sure the training data and the locked test set cover them.
2. Will you test on yourself only, or on other people, and how many?
3. Forced choice, or may the predictor say "not sure"? Do abstentions count as errors?
4. What should the screen show: the probability of the hand shown ("RIGHT 91 %") or always P(right)? See question e of [09 §10](09_STAGE2_3_PLAN.md#10-open-questions-for-the-pi).
5. Which odd movements should count? `chin` and `forehead` are not nose touches, and the tilted touch uses the phone's edge.
6. Is a 2.5-s upright check before testing acceptable? No table is needed.
7. Should predict-mode attempts be stored for later analysis? That needs consent text.
8. Is a per-user calibration allowed, i.e. a few labelled attempts before testing? It would change the model design ([05 §1](05_MODELING_PLAN.md#1-task-framing)); this plan assumes not.
