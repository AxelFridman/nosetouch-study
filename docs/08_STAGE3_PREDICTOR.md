# Stage 3: the predictor for the PI

This document covers what the PI will use to evaluate the work, how the Stage-2 model becomes an in-browser predictor, how to test it fairly, and the hand-off checklist.

Related: [Modeling plan](05_MODELING_PLAN.md) · [Protocol](01_PROTOCOL.md) · [Segmentation](04_SEGMENTATION.md) · interface [predictor.ts](../app/src/core/predictor.ts).

**Legend.** **Rec.** = recommendation; **Hyp.** = hypothesis. **Status:** Stage 1 implements only the `Predictor` interface. It ships no model on purpose, and everything else in this document is to be built in Stage 3.

---

## 1. What the PI uses

The PI uses the same PWA, from the same link or QR code, with no install. It runs in a **predict mode**. **Rec.:** enter predict mode with `?predict=1`, by analogy with `?sim=1`.

1. **Open the link** in Safari (iPhone) or Chrome (Android). In-app browsers such as Instagram, Facebook or an Android WebView are detected (`device.in_app_browser`) and the user is told to switch.
2. **Tap "Enable sensors".** This shows the motion permission prompt on iOS; Chrome 151+ also exposes `requestPermission()`.
3. **Device check.** It takes about 15 s and runs once per page load:
   - the phone lies flat on a table, which gives the accelerometer sign and the gyro bias;
   - a tilt gesture checks the rotation-rate axes (`axis_check`);
   - a live rate check requires at least 50 Hz.

   The screen states the result, e.g. "WebKit: accelerometer inverted, corrected".
4. **Trial.** The PI holds the phone at arm's length in any grip. The cues match the collector: Ready, hold still, GO beep, bring the phone to the nose, hold until the double beep.
5. **Result.**
   - Normally: **LEFT** or **RIGHT** in large type, `p_right` as a percentage, a confidence label and `model_id`. It also shows the evidence: net yaw about gravity, duration, and a mini plot with the detected onset and end.
   - If the predictor abstains: "Not sure, please repeat", with the reason.
6. **Optional scoring.** After the result, the operator taps the true hand. The app keeps a local log that can be exported as CSV.

The app never asks for the hand, grip, speed or variant: the model is blind to `condition` by construction (§2.4).

## 2. Architecture

### 2.1 Components

```
PredictScreen (new)
 └─ TrialController (unchanged; predict-mode segmentation params, placeholder condition)
     └─ TrialRecord (streams + online_v1 segmentation)
         └─ Predictor.predict(trial)   predictor bound to this page load's calibration + device info
              1 gates: quality, calibration ............ abstain reasons
              2 norm_v1: sign, axes, units (+ iOS quantisation if the model was trained with it)
              3 resample to 60 Hz
              4 gravity_v1: gyro propagation from the rest windows
              5 segmentation: same method and parameters as in training
              6 features: TS port of the deployed subset
              7 model: JSON or ONNX
              8 calibration: temperature or Platt
              9 abstention rules ........................ Prediction
```

Code layout (**Rec.**):
- The pipeline lives in `app/src/predict/` and is loaded with a dynamic `import()` only in predict mode, so the collector does not grow.
- Models live in `app/public/models/<model_id>.json` (or `.onnx`).
- `app/public/models/index.json` names the active model, so the model can be swapped without rebuilding the app.

### 2.2 Interface

The current contract in `app/src/core/predictor.ts`:

```ts
export interface Prediction { hand: Hand; p_right: number; model_id: string; }
export interface Predictor { readonly id: string; predict(trial: TrialRecord): Prediction; }
```

The proposed Stage-3 extension (**Rec.**) is additive:

```ts
export interface PredictionV2 extends Prediction {
  status: 'ok' | 'abstain';
  confidence: 'high' | 'medium' | 'low';
  reasons: string[]; // e.g. 'rate_below_50hz', 'no_onset', 'low_confidence'
  evidence: { net_yaw_deg: number | null; duration_ms: number | null };
  versions: { norm: string; gravity: string; segmentation: string; features: string };
}
export interface AsyncPredictor {
  readonly id: string;
  predict(trial: TrialRecord): Promise<PredictionV2>;
}
```

- **Why async.** In ONNX Runtime Web, `InferenceSession.create()` and `session.run()` both return Promises ([inference-session.ts](https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/inference-session.ts)). JSON models can stay synchronous behind the same interface.
- **Abstentions.** When the predictor abstains, `hand` still carries the arg-max, so a forced-choice score can always be computed.
- **Calibration context.** Normalisation needs the gravity sign, the axis check and the gyro bias, plus `device.engine` as a fallback. These live in `SessionRecord.calibration` and `SessionRecord.device`, not in `TrialRecord`. **Rec.:** bind them when the predictor is created, `createPredictor(model, { calibration, device })`, instead of widening `predict()`.

### 2.3 Model packaging

| Option | Models | Artefact | Runtime cost | Notes |
|---|---|---|---|---|
| JSON weights | `YawSignRule`, logistic regression (including the antisymmetric variant) | about 1–5 KB | none (a dot product) | The default. |
| JSON trees | `HistGradientBoostingClassifier` | tens to hundreds of KB (**Hyp.**: depends on trees × depth) | none (tree walks) | Written by a motionlab exporter that reads the fitted model's internal tree arrays. Pin the scikit-learn version. |
| ONNX + onnxruntime-web | GBM or logistic regression via skl2onnx, which lists `HistGradientBoostingClassifier` and `LogisticRegression` as supported ([skl2onnx 1.20](https://onnx.ai/sklearn-onnx/supported.html)); CNN or GRU via a PyTorch export | model ≤ 2 MB (**Rec.**) | The CPU binary `ort-wasm-simd-threaded.wasm` is 14.2 MB in onnxruntime-web 1.30.0 ([package files](https://app.unpkg.com/onnxruntime-web@1.30.0/files/dist)). | The WASM backend supports all `ai.onnx` and `ai.onnx.ml` operators ([README](https://github.com/microsoft/onnxruntime/blob/main/js/web/README.md)), and a custom build can shrink the runtime ([ORT web docs](https://onnxruntime.ai/docs/tutorials/web/)). Lazy-load it. |
| Hand-written TS forward pass | small 1D-CNN or GRU (≲ 50k parameters) | JSON weights, ≲ 200 KB | none | A few hundred lines of code. It avoids the 14 MB runtime, but needs a parity test against PyTorch. |

**Decision rule (Rec.).** Ship the simplest model whose locked-test accuracy lies inside the CI of the best model. A neural model justifies ONNX only if it beats the GBM by more than the CI width.

Fields every JSON model file carries (**Rec.**). The numbers below are placeholders, not results:

```json
{
  "model_id": "lr-chiral-v1",
  "kind": "logistic_antisymmetric",
  "versions": { "norm": "norm_v1", "gravity": "gravity_v1",
                "segmentation": "online_v1+predict_params_v1", "features": "feat_v1" },
  "resample_hz": 60,
  "quantise_ios": true,
  "feature_names": ["psi_n", "helicity", "accel_turning"],
  "scale": [1.0, 1.0, 1.0],
  "coef": [0.0, 0.0, 0.0],
  "temperature": 1.0,
  "abstain_below": 0.65,
  "training": { "data_snapshot": "sha256:...", "git_commit": "...", "n_participants": 0,
                "cv_accuracy_ci": [null, null, null], "locked_test_accuracy_ci": [null, null, null] }
}
```

### 2.4 Training and deployment parity

1. **Deployment rate.** Train the deployed model on 60 Hz PWA data. High-rate phyphox data serve only for validation, unless they are converted with the PWA emulation ([05 §2.2](05_MODELING_PLAN.md#22-resampling-rec)).
2. **Same normalisation.** `norm_v1` is implemented identically in TypeScript and Python, so an iPhone's input looks like the training data. That includes the iOS quantisation when the model was trained with it (`quantise_ios`; it is off by default until ablation A3 decides, [05 §2.1](05_MODELING_PLAN.md#21-normalisation-norm_v1)).
3. **Same segmentation.** The browser must reproduce the training boundaries exactly. **Rec.:** re-segment every training trial with online_v1 using the predict-mode parameters below, via `runOnlineSegmentation(samples, params, goT)` in TS or the motionlab port, which by spec gives identical indices. Alternatively, port the chosen offline method to TS.
4. **Same features.** Port only the deployed feature subset to TS, and keep that subset small.
5. **Golden-file parity.**
   - motionlab writes golden fixtures (input trial → features → probability) for about 50 trials, drawn from synthetic data and consented pilot data.
   - A vitest test requires feature differences within 1e-6 (relative).
   - It requires probability differences within 1e-6 for JSON models and within 1e-4 for ONNX (float32).
6. **Version check.** The predictor refuses a model whose `versions` differ from its own pipeline.
7. **Condition blindness.** A test randomises `trial.condition` and asserts that the prediction is unchanged.

**Predict-mode segmentation parameters (Rec.; validate on pilot data).** The speed and variant are unknown, so use the most permissive composition the protocol already defines: the values `resolveParams` returns for speed `slow` combined with variant `pause`.

| Parameter | Value |
|---|---|
| `onset_gyro_min_dps` | 15 |
| `onset_acc_min_ms2` | 0.3 |
| `end_gyro_dps` | 10 |
| `end_acc_ms2` | 0.25 |
| `end_sustain_ms` | 1800 |
| `max_movement_ms` | 12000 |
| all others | `DEFAULT_SEGMENTATION_PARAMS` |

- **Trade-off.** The person must hold the phone at the nose for about 2 s, but pauses shorter than 1.8 s no longer cut the movement short.
- **Placeholder condition.** `TrialController` still needs a `TrialCondition`. Predict mode passes a fixed placeholder that the predictor never reads (rule 7).
- **Keep predict-mode records separate.** They need a marker so they never mix with research data. **Rec.:** a distinct `study_code` such as `nose-touch-predict`, and no upload.

### 2.5 Budgets (Rec.)

| Item | Budget | Check |
|---|---|---|
| Growth of the collector bundle | 0 KB: predict code sits behind a dynamic import | `vite build` output |
| JSON model | ≤ 200 KB | file size |
| ONNX model | ≤ 2 MB, plus the lazy-loaded 14.2 MB runtime | network panel |
| Inference time | ≤ 50 ms for JSON models, ≤ 500 ms for ONNX, on a mid-range phone | `performance.now()` around `predict`, logged |
| From the end of the trial to the shown result | ≤ 1 s | manual test |
| Offline | works after the first predict-mode visit, with the service worker caching model and runtime | airplane-mode test |
| Hosting | far below the GitHub Pages limit of 1 GB per site ([limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)) | none |

## 3. An evaluation protocol the PI can run

### 3.1 Design (Rec.)

- **Phone.** Use the PI's own phone. Beforehand, state which results apply to it: the leave-one-platform-out and leave-one-device-model-out numbers ([05 §6](05_MODELING_PLAN.md#6-evaluation-protocol)).
- **People.** Use people who are not in the training data: at least 5, better 10. More people with fewer trials each gives tighter intervals than a few people with many trials (table below).
- **Trials.**
  - 10 per person, 5 with each hand.
  - The PI randomises the hand order, e.g. with a coin or a pre-printed sequence.
  - The PI picks the grips, including screen away, landscape and upside down.
  - Add 2–3 odd movements per person, e.g. curved, pause, twist.
- **Blinding.** The app does not know the hand, and the student neither chooses nor sees the hands during testing.
- **Recording.** The operator enters the true hand after each result. Score standard and odd movements separately.
- **Stress tests (optional, unlabelled).** Shake the phone, lift it to the ear, or put it down on the table. The desired behaviour is to abstain.

### 3.2 How to report

- **Report:**
  - accuracy on answered trials;
  - coverage, the share of trials answered;
  - accuracy with abstentions counted as errors (the conservative number);
  - accuracy per person;
  - Brier score.
- **Uncertainty.** With 5 or more people, use a participant-level bootstrap. Otherwise give the pooled count with an exact (Clopper–Pearson) interval, and state that it ignores the correlation between trials of the same person. With about 100 test trials, error bars of about ±10% are normal ([Varoquaux 2018](https://arxiv.org/abs/1706.07581)).

Exact 95% CIs for a pooled count (my computation):

| Correct / trials | 95% CI |
|---|---|
| 19 / 20 (95%) | 75–100% |
| 38 / 40 (95%) | 83–99% |
| 95 / 100 (95%) | 89–98% |
| 90 / 100 (90%) | 82–95% |
| 180 / 200 (90%) | 85–94% |

Trials of one person are correlated, so the effective n is n / (1 + (m − 1)·ICC), with m trials per person (the standard design effect). The table gives the half-width of a 95% CI at 90% accuracy (my computation):

| Plan | ICC 0.05 | ICC 0.1 | ICC 0.2 |
|---|---|---|---|
| 5 people × 20 trials | ±8 pts | ±10 pts | ±13 pts |
| 10 people × 10 trials | ±7 pts | ±8 pts | ±10 pts |
| 10 people × 20 trials | ±6 pts | ±7 pts | ±9 pts |

Report template:

| Field | Value |
|---|---|
| `model_id`, `app_version` | |
| Phone: platform, model, browser | |
| People / trials / answered | |
| Accuracy on answered trials (95% CI) | |
| Coverage | |
| Accuracy with abstentions as errors | |
| Standard vs odd movements | |
| Per-person accuracy (min–max) | |
| Expectation stated beforehand | |

### 3.3 State the expectation in advance (Rec.)

Before the PI's session, the student writes down the expected accuracy and its CI from the locked test set ([05 §6.4](05_MODELING_PLAN.md#64-locked-test-set-rec)), for the PI's platform and for standard vs odd movements. The session then tests a stated prediction instead of producing a number to explain afterwards.

## 4. Failure handling

Abstaining beats a confident wrong answer. Every abstention names its reason.

| Situation | Detection | Behaviour |
|---|---|---|
| No permission, or no motion events | sensor error `no_events` or `permission_denied` | Block; explain how to recover (Safari/Chrome settings). |
| In-app browser | `device.in_app_browser` | Warn; ask the user to open the page in Safari or Chrome. |
| Incomplete recording | `quality.motion_hz` < 50, `gaps_over_50ms` > 0, or `hidden_during` | Abstain: "recording incomplete". Ask the user to turn off Low Power Mode and keep the screen on. |
| No gyroscope | `quality.rotation_available` false | Refuse: the model needs ω. |
| Calibration missing or unclear | `gravity_sign` null, or `axis_check.verdict` `unclear` | Repeat the device check. If it stays unclear, use the engine prior and mark the result low confidence. |
| No movement / movement too long | `outcome.status` `no_movement` or `too_long` | Abstain. |
| Movement outside the training range | duration < 150 ms or above the training maximum; peak ‖ω‖ above the training maximum | Abstain. |
| Multi-peak or odd movement | more movement units than in training, or methods disagree by > 100 ms | Predict, with confidence at most "medium". |
| Out-of-distribution features | any feature outside the 0.5–99.5 percentile range of training | Warn; abstain if more than one feature is out of range. |
| Low confidence | max(p, 1 − p) < τ | Abstain: "not sure". |
| Mirror inconsistency (unconstrained models only) | \|p(x) + p(m(x)) − 1\| > 0.2 | Warn. |
| Possible clipping | ‖f‖ plateaus near the sensor range | Warn. |

- **Choosing τ (Rec.).** Pick τ from the development out-of-fold predictions, never from the locked test. Use the smallest τ for which accuracy on answered trials reaches the target (e.g. 95%), and report the coverage that τ gives on the locked test.
- **Forced choice.** If the PI wants forced choice, set τ = 0.5 and keep the confidence label.

## 5. Privacy in predict mode (Rec.)

- **No uploads in predict mode.** No profile is needed; an ephemeral `P-XXXXXX` code satisfies the schema.
- **Local scoring log.** The log stays on the phone, and export is manual.
- **Testing other people.** Ask whether the ethics approval covers a demonstration. Storing predictions for later analysis counts as data collection and needs the collector's consent flow.

## 6. Hand-off checklist

1. The locked-test report is final, and the expected accuracy per platform is written down (§3.3).
2. The model file in `app/public/models/` carries its versions, the data snapshot hash, the git commit and its metrics with CIs.
3. The golden-file parity test (TypeScript vs Python) passes.
4. The condition-blindness test passes.
5. Refusal on a version mismatch is tested.
6. Predict mode works on iOS Safari and Chrome Android, on at least 2 phones each and on the PI's model if known. This includes the device check, both gravity signs, and Rotation Lock on iOS, since iOS cannot lock the screen orientation ([MDN BCD](https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json)).
7. The offline test passes: airplane mode after the first load.
8. Each abstention reason has been triggered once in a test: no movement, too long, low rate, low confidence, missing calibration.
9. The latency and size budgets (§2.5) are measured and recorded.
10. Predict mode makes no network requests after loading, verified in the browser's network panel or with Playwright.
11. A one-page sheet for the PI covers the QR code, the steps, the scoring sheet, what counts as an error, and the known limitations.
12. Known limitations are listed, e.g. odd movements, slow movements from a `center` start, phones without a gyroscope, in-app browsers, and browsers below 50 Hz.
13. The previous model is kept and can be selected in `models/index.json` (rollback).
14. `model_id` and `app_version` are visible on the result screen.
15. A desk demo with `?sim=1&predict=1` works. This is a known-answer check only, not evidence of accuracy.

## 7. Open questions for the PI

1. Which phone or phones will you test with (platform, model)? We need to make sure the training data and the locked test set cover them.
2. Will you test on yourself only, or on other people, and how many?
3. Forced choice, or may the predictor abstain? Do abstentions count as errors?
4. Should the result show `p_right`, or only LEFT or RIGHT?
5. Which odd movements should count? `chin` and `forehead` are not nose touches.
6. Is a device check of about 15 s at the start acceptable?
7. Should predict-mode trials be stored for later analysis? That needs consent.
8. Is a per-user calibration allowed, i.e. a few labelled trials before testing? It would change the model design ([05 §1](05_MODELING_PLAN.md#1-task-framing)).
