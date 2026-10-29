# Stage 3: the production page — app-side spec

The production page is a predict mode of the same PWA. It is cue-free: the person brings the phone to the nose once or several times, without being told when to start. The page finds each touch itself and shows the calibrated **P(right)**, computed from sensor data only. It also decides when the answer is final.

- **Link:** <https://axelfridman.github.io/nosetouch-study/?predict=1> (alias `?mode=predict`). It works once the predict code is deployed to `main`.
- **Options:** `&model=cnn|m4|phys6|yaw`, `&view=compare`, `&lang=he`, `&debug=1`. The desktop demo is `&sim=1`, with optional `&hand=left|right` and `&seed=<n>`.
- **Contracts:** [`app/src/predict/types.ts`](../../app/src/predict/types.ts), which is type-only. Its section numbers match this document.
- **Python ground truth:** `analysis/stage2/lanes/m4_production.py` (`predict_stream`) together with `m4_detect`, `m4_e2e.FittedPipeline.attempts`, `m4_online` and `stage2/standardize.py`, and `final_infer.py` for the final CNN. The TypeScript is a port of these files, not a redesign.
- **Replaces** the cue-based [08_STAGE3_PREDICTOR.md](../08_STAGE3_PREDICTOR.md). That document's routing, packaging, permission handling, privacy rules and colours are kept; its GO cue, upright check, TrialController and online_v1 are dropped.

---

## 1. Sensors: from a DeviceMotion event to the stream the models read

### 1.1 What the collector does today

| Step | Code | What happens |
|---|---|---|
| Capture | `core/sensors.ts` `createBrowserSensorSource()` | Listens to `devicemotion`. Each event becomes a `MotionSample`: `t = performance.now()` (first statement of the handler, ms), `ts = event.timeStamp`, `acc = acceleration`, `accg = accelerationIncludingGravity` (m/s²), `rr = rotationRate {alpha, beta, gamma}` (deg/s, alpha about device x, beta about y, gamma about z: W3C 2025), and `interval`. A vector with a missing or non-finite component becomes `null`. Values are **raw**: no sign fix, no rounding. If no data-carrying event arrives within 1.5 s, `onError('no_events')` fires (desktop Chrome sends events with all fields null). |
| Permission | `requestMotionPermission()` | Must be the **first statement of the tap handler**, with no `await` before it. It calls `DeviceMotionEvent.requestPermission()` and `DeviceOrientationEvent.requestPermission()` synchronously wherever they exist (iOS 13+, Chrome 151+) and returns `granted / denied / unknown / not_required`. iOS keeps the decision in memory until Safari is force-quit, and a Cancel sticks just as long. |
| Engine | `getDeviceInfo().engine` | `'webkit'` exactly when `webkitCompassHeading` is on `DeviceOrientationEvent.prototype` (every iOS browser), otherwise `gecko`, `blink` or `unknown`. `in_app_browser` detects Instagram, Facebook, WhatsApp and the other in-app browsers. |
| Storage | `core/recorder.ts` `StreamRecorder.addMotionRel()` | Columns `t, ts, acc_x/y/z, accg_x/y/z, rr_alpha/beta/gamma` of `TrialRecord.streams.motion`. `t = round(t − t0, 2)`, acc/accg rounded to 4 decimals, rr to 3, ts relative to the first raw timeStamp, `interval_raw` = the first finite interval. null stays null. |
| Sign and axes | `motionlab/conventions.py` norm_v2 | WebKit reports `acc` and `accg` **negated on every axis**; rotationRate signs agree across engines. The session sign comes from the calibration, or from the engine prior confirmed by trial votes. In v2 the calibration is implicit (`implicit_v1`, from the first still upright trial), there is no table step and no gyro bias. The rate mapping is W3C deg/s, checked by the gyro-gravity closure. |

### 1.2 What `stage2.standardize.stream_from_record` needs

The production path is `stream_from_record(record, norm)`, then `Stage2Data.stream(i)`, then `m4_detect.as_stream` (which drops the gravity magnitude and the baseline gravity). It needs:

1. `record.streams.motion`: `t` and the nine value columns (`ts` and `interval_raw` are not read);
2. from `norm`: the **gravity sign** (multiplies acc and accg), the **rotationRate mapping and units**, the **engine** (the repeated-sample rule is stricter on WebKit; `quantize_ios` is off), and nothing else on the production path.

The steps are: drop isolated repeated samples → interpolate linearly onto a uniform 60 Hz grid starting at the first kept sample (gaps over 100 ms become NaN, `valid = 0`) → zero-phase Butterworth low-pass (order 4, 15 Hz, `sosfiltfilt`, padlen = min(15, run − 1), run by run).

### 1.3 A live buffer that is the same as a recorded trial

`pipeline/liveBuffer.ts` (`LiveBuffer`, types §7) is required to:

- **round exactly like `StreamRecorder.addMotionRel`**, using `roundValue` and `ROUNDING` imported from `core/recorder.ts` and `core/constants.ts`, never copied. `t0` is the first sample's `t`, so the stream starts at `t = 0`.
- produce `raw(n)`, which equals `rawMotionFromRecord(record.streams.motion)` of the record the collector would have stored for the same samples. A test feeds identical samples to both and compares the two.
- reject a sample whose `t` is not after the previous one. The recorder keeps such samples; the browser never produces them.

**Production normalisation** (`productionNormalization(engine)`): gravity sign −1 for `webkit`, +1 for `blink`, `gecko` and `unknown`; mapping `w3c`; `deg_per_s`; `quantizeIos: false`. There is no calibration step and no gyro bias.

This is safe for two reasons:
- It is norm_v2's normal case: the engine prior confirmed by the trial votes is the sign source `engine_vote` in `motionlab/conventions.py`.
- A wrong sign negates gravity, so the phone looks as if its screen faced away. The grip gate (§4.4) then refuses to answer instead of inverting the result.

### 1.4 Reuse and non-reuse

Import these directly; never copy them.

| Piece | From | How predict mode uses it |
|---|---|---|
| Capture | `core/sensors.ts` `createBrowserSensorSource({quaternion: false, absoluteQuaternion: false, orientationAbs: false, magnetometer: false, screenEvents: false})` | `onMotion` → `engine.push`, `onOrientation` is a no-op, `onError` → `SourceFailure` |
| Permission | `requestMotionPermission()` | first statement of the Start tap |
| Preflight | `detectCapabilities()`, `getDeviceInfo()` | secure context, DeviceMotionEvent, in-app browser, engine, desktop |
| Wake lock | `core/cues.ts` `createWakeLockGuard()` | acquired in the Start tap, released on Reset and dispose |
| Automation check | `isAutomatedBrowser()` | no vibration under Playwright |
| Rounding | `core/recorder.ts` `roundValue`, `rateStats`; `core/constants.ts` `ROUNDING`, `APP_VERSION` | LiveBuffer; footer |
| Simulator | `sim/simulatedSource.ts`, `sim/model.ts` | the `&sim=1` demo (§3) |
| Design tokens | `ui/styles.css` (already global) | colours, font, radius, tap size, dark mode |

**Never imported into predict mode:** `core/storage` (IndexedDB), `core/uploader`, `core/config` (fetches `config.json`), `createCueSink` (audio and speech: predict mode makes no sound), `core/trialController`, `core/segmentation` (online_v1, cue-based), `core/calibration`, `core/protocol`, `core/exporter`, and everything under `app/` (`services`, `store`, `sessionFlow`, screens). A test greps the predict sources for these imports.

---

## 2. Boot, routing, service worker, build and deploy

### 2.1 `main.tsx` (UI owner; the only collector file that changes)

```tsx
import { render } from 'preact';
import { App } from './app/App';
import './ui/styles.css';

const root = document.getElementById('app')!;
const q = new URLSearchParams(location.search);
if (q.get('predict') === '1' || q.get('mode') === 'predict') {
  import('./predict/mount').then(
    (m) => m.mount(root, location.search),
    () => { root.textContent = 'Could not load. Check the connection and reload.'; },
  );
} else {
  render(<App />, root);
  if (import.meta.env.PROD && 'serviceWorker' in navigator) { /* unchanged registration */ }
}
```

- **The collector is untouched.** `App` and `styles.css` stay static imports, so the collector chunk and its behaviour do not change. In predict mode their modules are evaluated but never run: they have no import-time side effects (checked: only module-level `Map`/`Set` caches), and `boot()` runs inside `App`'s effect, so there is no `config.json` fetch, no IndexedDB and no uploader.
- **`mount(root, search)`** (`predict/mount.tsx`) parses `PredictParams` and inserts `<meta http-equiv="Content-Security-Policy" content="connect-src 'none'">` (production builds only, because the dev server's HMR WebSocket needs it). That blocks fetch, XHR, WebSocket and sendBeacon for the life of the page. It also sets `lang` and `dir` on `<html>` and renders `PredictApp`.
- **No `history` routing.** Predict mode is a single page. The model selector updates `&model=` with `history.replaceState`, so a copied link pins the model.

### 2.2 Service worker (`public/sw.js`, unchanged)

- **No registration in predict mode.** Registering fetches `sw.js`, and its install pre-caches `config.json`, which is a network call that predict mode must not make.
- **An existing worker still helps.** If an earlier collector visit installed it, it controls `?predict=1` too. Navigations are network-first with the cached `index.html` as fallback, and `./assets/*` is cache-first and cached at runtime, which includes the predict chunk. Predict mode therefore works offline after one online predict visit on a phone that has visited the collector. Otherwise offline is not supported, which is acceptable: the PI tests online.

### 2.3 Build (`vite.config.ts`, unchanged)

- Rollup splits `import('./predict/mount')` into its own chunk, `assets/mount-<hash>.js`, plus a CSS chunk if `predict.css` is imported.
- The model JSON files are statically imported inside `predict/models/`, so they live in that chunk. There is no runtime fetch.
- `base: './'` keeps the chunk URLs relative (they work under `/nosetouch-study/`), and `target: 'es2020'` covers iOS 14+.

**Budgets.**
- The entry chunk grows only by the branch, the no-network policy and Vite's preload helper: measured +2.4 kB raw, about 1 kB gzip (352.06 kB at 8eec3d4 to 354.44 kB at 0.3.0).
- The predict chunk is at most 2.5 MB raw and 1 MB gzip. `final_model.json` alone is 1.6 MB of text. Keep its exact decimal strings: parity needs the same float64 values that `final_infer.py` reads.

**Integrity.** The JSON is transformed into a JS module, so its original bytes cannot be hashed at runtime. Instead, `models/registry.ts` stores the sha256 of each file, and a vitest test recomputes the hashes from the files on disk. The footer shows the first 8 characters.

A post-build test runs when `dist/` exists:
- no `assets/index-*.js` contains `nosetouch-final-1`;
- a `mount-*.js` chunk does.

### 2.4 Deploy (`.github/workflows/deploy.yml`, unchanged)

A push to `main` that touches `app/**` runs `npm ci`, `npm test` and `npm run build`, then uploads `app/dist` to GitHub Pages. Predict mode needs nothing more. A CI failure in the predict tests therefore blocks the deploy, which is intended.

Suggested release: app version `0.3.0`. `APP_VERSION` in `core/constants.ts` and `package.json` must change together; a test checks this.

---

## 3. Desktop demo: `?predict=1&sim=1`

This is a known-answer check, not evidence of accuracy. It goes through the same live path as the real sensors: `LiveSample` → `engine.push` → UI.

`live/demoSource.ts` implements `LiveSource` with `kind: 'demo'`.

1. **Script, generated at Start.** It is synchronous and seeded by `&seed`, or by a random seed.
   - Hand `H` = `&hand`, or random. Platform: `defaultSimPlatform('android' | 'ios', rng)`; the demo's `engine` is that platform's (`ios` → `webkit`, gravity sign −1).
   - Participant: `randomSimParticipant(rng)`. Condition `{hand: H, orientation: 'upright_toward', speed: 'natural', start_position: 'natural', variant: 'standard', posture: 'standing', rep: 0}`.
   - Six touches, each one `createSimulatedSource({mode: 'offline', startTime: <end of the previous one>, ...})`.
2. **Driving each touch.** The listener calls `notifyCue('go', t)` once the source has been still for 1.2–2.0 s, and `notifyCue('end', t)` 0.4–0.9 s after `truth.end_t`. It stops the source 1.0–1.5 s after `truth.return_end_t`.
3. **Samples.** Motion samples are kept and orientation events are dropped.
4. **Replay.** A 60 Hz `setInterval` emits every sample whose `t − t_first` is at most the elapsed wall time, with `t` rebased onto `performance.now()`.
5. **Page behaviour.**
   - No permission prompt, wake lock or vibration, and never sound.
   - A badge reads "Demo: simulated LEFT hand".
   - Reset generates a new script: the same hand when `&hand` is given, otherwise a new random draw.

**Risk.** The segment classifier was fitted on real movements. If simulated approaches are not accepted as attempts, the demo abstains. The demo e2e test (§4.2) shows this at once. If it happens, the demo uses a committed synthetic stream exported from `analysis/motionlab/synth.py` and accepted by the Python detector. It never uses a real recording.

---

## 4. The live engine: cue-free and continuous (detect owner)

### 4.1 The pure function

`predictStream(raw, norm, model, detector)` ports `m4_production.predict_stream`. The steps:

1. `streamFrom`;
2. `segmentRows`: bouts → `window_for` (crop [onset − 1 s, end + 0.5 s]) → features → P(approach / return / other) → the attempt rule (arg-max is 'approach', P(approach) ≥ `det_threshold`, canonical OK, no pairing);
3. `rest_up_z` of the stream: the still samples before the first bout, or the first 0.25 s of the stream;
4. the grip gate (below the page's gate, -0.05 → status `grip`, P = 0.5; lane m4's `GRIP_GATE_UPZ` is 0, see `analysis/stage3/runtime.py` `PRODUCTION_GRIP_GATE_UPZ`);
5. `stopping_rule` over the attempts in decision order: P_k by the model's own aggregation; answer as soon as max(P_k, 1 − P_k) ≥ 0.9; after 5 attempts, answer if max ≥ `p_answer` (0.5), otherwise abstain.

Parity tests compare this function with Python on complete streams.

### 4.2 The live loop (`createSessionEngine`)

The loop is driven only by sample times, so a replay of the same samples gives the same snapshots.

1. **Buffer.** `push` appends to the `LiveBuffer`. Each time stream time crosses a multiple of `tickMs` (100 ms), it runs one tick synchronously: `streamFrom(buffer.raw())` → activity → bouts. All of this is O(n): a 3-minute buffer is 10,800 samples, which should take a few ms. Measure it on a phone, and in the debug panel.
2. **Confirm and freeze.** A bout is confirmed when three things hold:
   - it has ended (a stillness of at least `end_sustain` follows it);
   - the buffer holds its full look-ahead (iEnd + nLook − 1 ≤ n − 1);
   - the stream extends `settleMs` = 1000 ms past its `decidedMs`.

   Each confirmed bout is processed once and frozen into an `AttemptResult` (every model's output) or an `IgnoredMovement`. A frozen bout is never recomputed; a later tick's bout that overlaps it is the same bout.
3. **Why the settle margin.** The zero-phase low-pass of a growing buffer changes the last samples. For a gyro-like signal, the difference between filtering a prefix and filtering the full recording is about 5e-5 at 30 samples before the prefix end, 1e-6 at 40 and 3e-10 at 60 (scipy, measured). Everything a bout's window, features and rest windows depend on lies at or before `decidedMs`. After 1 s of settling, the live values therefore equal those of the complete recording to well under 1e-6.
   - **What this guarantees.** The live session gives the same attempts and the same decision as `predict_stream` on the whole recording, keeping its attempts with `decidedMs + settleMs ≤ t_last`.
   - **No UX cost.** At that moment the phone is at the nose or on its way back, so nobody can read the screen yet.
4. **Decide.** After each new attempt, `stoppingRule(driver, attempts, streamEnded=false)` runs.
   - Not final → phase `watching` with a provisional P.
   - `confident` or `k_max` → phase `final`. The engine stops reading samples until `reset()`. The sensors keep running, so Again needs no new permission.
   - Stream time reaches `maxStreamMs` (180 s) → final with `streamEnded=true`: `stream_end` if any attempt exists, otherwise `no_attempt` (abstain).
5. **Grip.** `restUpZ` is final once the first bout is frozen. Below the gate → phase `grip`, not answered. The user taps Start over, which makes a new stream.

   **Hint.** While watching, `screenAwayHint` is set when the phone has been still for 0.5 s with median up_z < 0. It is UI only and never changes a result.
6. **Model switch.** `setDriver(id)` recomputes the decision from the frozen attempts immediately (`streamEnded` = the session is already final). Compare-all shows every model after the same `nUsed` attempts, each with its own calibrator and aggregation.
7. **Gaps.** When the page is hidden or the sensor stalls for more than 1 s, the UI calls `notifyGap()`: while watching the engine resets the stream; a final answer stays.

**Constants** (engine options, defaults in `detect/engine.ts`):

| Constant | Value |
|---|---|
| `tickMs` | 100 |
| `settleMs` | 1000 |
| `maxStreamMs` | 180 000 |
| hint window | 500 ms |
| sample checks | the first 1 s (`no_gyroscope`, `no_accelerometer`) |
| rate badge | below 50 Hz in `&debug=1` |

The detector parameters and the stopping thresholds (`p_stop` 0.9, `k_max` 5, `p_answer` 0.5) come from the exported m4 bundle (`DetectorBundle`). The code must say that **they were fitted for lane m4's classifier**: lane prod, which would refit them for the final CNN, has not run.

### 4.3 The models (models owner)

Every model reads the same `ApproachWindow`. That window is (17, 64) float32, exactly as `time_normalised` casts it.

| id | Stage 2 name | Input → logit | Calibrator | k-aggregation |
|---|---|---|---|---|
| `cnn` (default) | `m3_cal_deep_platt` (`final_model.json`) | + 4 body-frame channels; 12 channels → 4-member CNN, mean logit | Platt (a, b), logit clip 40, p clip 1e-4 | sigmoid(Σ z / τ_k), τ_1..τ_5 |
| `m4` | `m4_cal_nested` | yaw8: `yaw_cum` at samples 0, 8, …, 56 → logistic | the nested pick (none / temperature / isotonic_sym) | `scaled_mean` (s_k, or the plain mean when the declared rule chose it) |
| `phys6` | `m1_lr_phys6_nested` | yaw, roll and lateral velocity at 50 % and 100 % of the approach → antisymmetric logistic | its fitted calibrator | its own; exporter's choice |
| `yaw` | `example_yaw_logreg` | yaw8 (the reference) | its own | its own |

All four are fitted on **all persons** (P01–P20, `CAP_PER_PERSON` 40), with the same code paths as the evaluation. They are exported as JSON into `app/src/predict/models/json/`.

### 4.4 The screen check of the final model

`final_model.json` can un-flip a screen-away window when the rest pose before the movement has up_z below 0. **The app does not use this:** production asks the user to turn the phone (decision 2). So `AttemptInput.restUpZBefore` is not passed, and the app equals `final_infer.predict_proba(window)`. The un-flip path is still ported, and a parity test covers it.

---

## 5. UI design (UI owner)

**Principles.** Mobile-first, one column (max-width 480 px), safe-area padding, big type, one primary button, little text.
- Light and dark themes come from the collector's tokens (`--bg`, `--text`, `--font`, `--radius`, `--tap` = 52 px).
- **Hand colours: LEFT `#D1495B`, RIGHT `#00739A`.** They apply to the bar and to a small swatch only. Text stays in `--text` for contrast in dark mode, and the hands are always named in words.
- No sound, ever. Android vibrates once for 30 ms per attempt, never under automation.
- The P region is `aria-live="polite"`. The bar is `role="img"` with the label "LEFT 13 %, RIGHT 87 %".
- The bar stays left-to-right in Hebrew: LEFT and RIGHT are physical sides.

### 5.1 The probability display (`ProbabilityDisplay` + `HandBar`)

```
          P(right) = 87 %
  probability that the RIGHT hand was used
  LEFT ▕██████▏████████████████████▕ RIGHT        bar: left part (1 − p) #D1495B,
       13 %          ¦          87 %                    right part p #00739A, tick at 50 %
            RIGHT hand · 2 touches
```

- **Number format.** `percentRight` + NBSP + `%`. At the extremes it shows `>99 %` (p ≥ 0.995) or `<1 %` (p ≤ 0.005); "100 %" never appears.
- **Main line.** "P(right) = …" always refers to the RIGHT hand, whichever hand is more likely. The subtitle states what P means.
- **Provisional.** While watching after k ≥ 1 attempts, the display is muted (`--muted`, smaller) with "after 1 touch · not final".
- **Footer line** (all screens, small and muted): `cnn · 1a2b3c4d · v0.3.0`, plus `demo` when sim is on. A "?" opens a three-line info sheet.

### 5.2 Screens (`PredictScreen`; root `data-testid="pm-<screen>"`)

```
start                               permission                     watching (no attempt yet)
┌────────────────────────────┐      ┌───────────────────────────┐  ┌────────────────────────────┐
│ NoseTouch           CNN ▾  │      │ NoseTouch          CNN ▾  │  │ NoseTouch           CNN ▾  │
│                            │      │                           │  │                            │
│        Which hand?         │      │   Allow motion access     │  │            ◉               │
│                            │      │          (spinner)        │  │  Hold the phone out, still.│
│ Hold the phone at arm's    │      │                           │  │  Then touch your nose.     │
│ length, screen towards you.│      │                           │  │        ○ ○ ○ ○ ○           │
│ Then touch your nose.      │      │                           │  │                            │
│ [         Start          ] │      │                           │  │        Start over          │
│ cnn · 1a2b3c4d · v0.3.0    │      │ cnn · 1a2b3c4d · v0.3.0   │  │ cnn · 1a2b3c4d · v0.3.0    │
└────────────────────────────┘      └───────────────────────────┘  └────────────────────────────┘

watching (after an attempt)         answer (final)                 abstain / grip
┌────────────────────────────┐      ┌───────────────────────────┐  ┌────────────────────────────┐
│     P(right) = 64 %        │      │       P(right) = 87 %     │  │        Not sure            │
│ probability that the RIGHT │      │ probability that the RIGHT│  │     please repeat          │
│ hand was used              │      │ hand was used             │  │                            │
│ LEFT ▕███▏██████▕ RIGHT    │      │ LEFT ▕██▏████████▕ RIGHT  │  │ [         Again          ] │
│ after 1 touch · not final  │      │   RIGHT hand · 2 touches  │  ├────────────────────────────┤
│   ● ○ ○ ○ ○  Touch again.  │      │ [         Again         ] │  │   (phone turning icon)     │
│        Start over          │      │    Compare models ▸       │  │ Turn the screen towards    │
└────────────────────────────┘      └───────────────────────────┘  │ your face                  │
                                                                   │ [       Start over       ] │
unsupported                                                        └────────────────────────────┘
┌──────────────────────────────────────────────┐
│ This page needs a phone with motion sensors. │   one message per SourceFailure, plus
│ [  Try the demo  ]                           │   a button: Try the demo (desktop),
└──────────────────────────────────────────────┘   Copy link (in-app browser), Reload
```

- **Attempt detected.** Dot k fills with a check, plus the 30 ms vibration. The P display updates at the same moment (the engine freezes the attempt and updates P together).
- **Hints while watching.**
  - `screenAwayHint` shows "Turn the screen towards your face" under the dot.
  - The dot pulses while `still` is false.
- **Debug (`&debug=1`).** A table of attempts and ignored movements (bounds, P(approach), each model's p), the rate, `restUpZ` and the closure.
- **Model selector.** A chip in the top bar ("CNN ▾") opens a bottom sheet:
  - four radio rows, each a label plus one line: **Final CNN** (default; "1D CNN, 4 members"), **Detector model** ("lane m4, yaw logistic"), **Physics logistic** ("PHYS6: yaw, roll, lateral"), **Yaw reference** ("logistic on yaw, 8 instants");
  - a "Compare models" switch.

  The selection calls `engine.setDriver` and `replaceState(&model=)`.
- **Compare view.** Rows of model name, P(right) and a mini bar, "after k touches", and a check where that model's own rule is confident. Tapping a row selects that model.

### 5.3 Strings

| key | English | Hebrew (`&lang=he`) |
|---|---|---|
| title | Which hand? | באיזו יד? |
| instruct | Hold the phone at arm's length, screen towards you. Then touch your nose. | החזיקו את הטלפון ביד מושטת, המסך מולכם. ואז געו איתו באף. |
| start | Start | התחלה |
| permission | Allow motion access | אשרו גישה לחיישני התנועה |
| denied_ios | Motion access was refused. Close Safari completely, open the link again and tap Allow. | הגישה לחיישני התנועה נדחתה. סגרו את Safari לגמרי, פתחו שוב את הקישור ואשרו. |
| denied_other | Motion access was refused. Allow it in the browser's site settings and reload. | הגישה לחיישני התנועה נדחתה. אשרו אותה בהגדרות האתר בדפדפן וטענו מחדש. |
| watch | Hold the phone out, still. Then touch your nose. | החזיקו את הטלפון מושט, בלי לזוז. ואז געו באף. |
| again_hint | Touch again. | געו שוב באף. |
| p_line | P(right) = {p} | P(right) = {p} |
| p_meaning | probability that the RIGHT hand was used | ההסתברות שהשתמשו ביד ימין |
| left / right | LEFT / RIGHT | שמאל / ימין |
| verdict | {HAND} hand · {k} touches (1: "1 touch") | יד {HAND} · {k} נגיעות (1: "נגיעה אחת") |
| provisional | after {k} touches · not final | אחרי {k} נגיעות · לא סופי |
| abstain | Not sure – please repeat | לא בטוח – נא לחזור על התנועה |
| grip | Turn the screen towards your face | סובבו את המסך אל הפנים |
| again / start_over | Again / Start over | שוב / להתחיל מחדש |
| compare | Compare models | השוואת מודלים |
| demo_badge | Demo: simulated {HAND} hand | הדגמה: יד {HAND} מדומה |
| u_desktop | This page needs a phone with motion sensors. | הדף צריך טלפון עם חיישני תנועה. |
| u_inapp | Open this link in Safari or Chrome. | פתחו את הקישור ב-Safari או ב-Chrome. |
| u_https | This page needs a secure (https) link. | הדף צריך קישור מאובטח (https). |
| u_gyro | This phone reports no gyroscope. | הטלפון הזה לא מדווח על ג'ירוסקופ. |
| u_events | No sensor data. Reload the page. | לא מתקבלים נתוני חיישנים. טענו את הדף מחדש. |
| try_demo / copy_link / copied | Try the demo / Copy link / Link copied | להדגמה / העתקת הקישור / הקישור הועתק |
| restarted | Restarted: the page was hidden. | התחלנו מחדש: הדף הוסתר. |
| info | P(right) is the model's calibrated probability that the right hand was used. Hold the phone with the screen towards you. Nothing leaves this phone. | P(right) היא ההסתברות המכוילת של המודל שהשתמשו ביד ימין. החזיקו את הטלפון כשהמסך מולכם. שום מידע לא יוצא מהטלפון. |

`P(right)`, numbers and model ids are wrapped in `<bdi dir="ltr">` in Hebrew.

---

## 6. Module layout and ownership (`app/src/predict/`)

```
types.ts                 shared contracts (this spec); changes go through the orchestrator
mount.tsx                UI   entry of the chunk: params, CSP meta, lang/dir, render
PredictApp.tsx           UI   screen state machine: preflight → start → permission → engine snapshots
params.ts  i18n.ts       UI   URL parsing (PredictParams), en/he strings (§5.3)
live/browserSource.ts    UI   LiveSource over core/sensors (+ the first-second gyro/accg checks)
live/demoSource.ts       UI   LiveSource over sim/ (§3)
live/preflight.ts        UI   Preflight from detectCapabilities + getDeviceInfo
components/*.tsx         UI   ProbabilityDisplay, HandBar, AttemptDots, ModelSheet, CompareTable, Footer
predict.css              UI   scoped under .pm (no collector selector changes)
pipeline/                PIPELINE  liveBuffer, record (rawMotionFromRecord), normalize, preprocess
                                   (repeats, resample, sosfiltfilt with scipy's padding and zi, gaps),
                                   gravity (gyro_gravity), standardize (streamFrom, canonical,
                                   timeNormalised, cropStream); PipelineApi
detect/                  DETECT    activity, bouts (find_segments), features (segment_features +
                                   ZUPT kinematics), segmentClassifier, rows (segment_table + attempt
                                   rule), grip (rest_up_z), stopping, predictStream, engine; DetectApi
models/                  MODELS    registry (specs, sha256, loadModels, detector()), finalCnn
                                   (body channels, conv/BN/ReLU, bins), logistic (yaw8, PHYS6 features),
                                   calibrators, aggregate; ModelsApi
models/json/             MODELS    final_model.json (copied byte for byte), m4_production.json (hand
                                   model + DetectorBundle), m1_phys6.json, ref_yaw8.json
```

**Order and dependencies.**
- `pipeline` depends on nothing in `predict/`.
- `detect` depends on `pipeline` and on `models` for `HandModel` and `detector()`.
- The UI depends on `detect.createSessionEngine`, `models.specs` and `live/`.

**Working in parallel.**
- The UI can start against a fake `SessionEngine` built from `types.ts`.
- `detect` can start against `PipelineApi` stubs fed by fixture streams.
- `models` can start from the windows in the fixtures.

**Implementation rules.**
- Use filter coefficients exported from scipy; never design filters at runtime.
- Use float64 everywhere except the `Float32Array` window.
- Round percentages half-to-even, as Python's `round` does.
- Never use `Math.random` in the pipeline.

---

## 7. Tests

### 7.1 Vitest (`npm test`; `app/tests/**/*.test.ts`, Node environment, 20 s timeout)

Tests live in `app/tests/predict/*.test.ts`.

**Fixtures.**
- Committed and synthetic only, in `app/tests/fixtures/predict/*.json.gz`. They are loaded with `readFileSync(new URL('../fixtures/predict/<f>', import.meta.url))`, `gunzipSync` and `JSON.parse`.
- Golden files from **real** recordings are written only to `analysis/data/stage3_fixtures/` (git-ignored). The tests reach them with `new URL('../../../analysis/data/stage3_fixtures/', import.meta.url)` inside `describe.skipIf(!existsSync(...))`, so CI runs without them.
- The Python side writes both from the same exporter. Each case holds:
  - `normalization` and `motion` (the `streams.motion` columns);
  - `expected`: per-channel checksums and sample values of the stream, the activity, `bouts`, `segments` (features and probabilities), the attempt windows, each model's `zRaw / p / z`, P_k, the `predict_stream` dict, and `rest_up_z`.

**Checks.**

| Area | What is compared | Tolerance |
|---|---|---|
| Pipeline | the stream (f, gyro, acc, valid), canonical channels, windows | 1e-6 |
| Pipeline | `LiveBuffer` against `StreamRecorder` | exact |
| Detect | bouts | identical indices |
| Detect | features, segment probabilities | 1e-6 |
| Detect | attempts, `predictStream` decisions and status | identical |
| Detect | grip | identical |
| Models | logits and calibrated p of every model | 1e-6 |
| Models | the CNN against `final_model_check.json` (P per cached real window, so it runs only under `skipIf` with the real-window fixtures) | 1e-6 |
| Models | un-flip (§4.4), aggregation per k, registry sha256 against the files | tested |

**Engine.**
- Replaying a fixture's samples in random chunk sizes gives the same snapshots.
- The frozen attempts equal Python's attempts with `decidedMs + 1000 ≤ t_last`, to 1e-6, and the decision is identical.
- `setDriver`, `reset`, `notifyGap`, `maxStreamMs` and grip are tested.

**UI.**
- `params` parsing.
- The import guard of §1.4.
- The percentage formatting (`<1 %`, `>99 %`, half-to-even).
- The `mount()` CSP meta.

### 7.2 Playwright (`npx playwright test`; `e2e/predict.spec.ts`; Pixel 7 emulation on Chromium; dev server)

The spec imports `test` and `expect` from `./fixtures`; `tests/e2eHarness.test.ts` enforces this. Through that fixture, every request to another host fails the test and the page can make no sound.

1. **Demo, known answer.** `/?predict=1&sim=1&hand=right&seed=7`: tap Start and wait for `pm-answer`. Expect the text `P(right) = \d+ %`, a value above 50, and "RIGHT hand". Also check:
   - no request for `config.json` (via `page.on('request')`);
   - `indexedDB.databases()` has no `nosetouch` database;
   - the silence log is `{spoken: [], audioContexts: 0}`.

   Repeat with `hand=left`.
2. **Injected DeviceMotion events** (the real-sensor path). An `addInitScript` sets `DeviceMotionEvent.requestPermission = () => Promise.resolve('granted')` (the iOS flow). For the WebKit sign it also defines `webkitCompassHeading` on `DeviceOrientationEvent.prototype` and negates acc and accg. After the Start tap, a `page.evaluate` loop replays a committed synthetic stream at 60 Hz:

   ```js
   window.dispatchEvent(new DeviceMotionEvent('devicemotion', {
     acceleration, accelerationIncludingGravity, rotationRate: {alpha, beta, gamma}, interval: 16,
   }))
   ```

   `page.clock` can fast-forward it: it fakes `performance.now()` consistently. Expect the fixture's hand and P equal to the TypeScript `predictStream` on the same fixture.
3. **Unsupported.** With no events injected, `pm-unsupported` appears with the no-events message after 1.5 s, and "Try the demo" leads to `&sim=1`.
4. **Layout and language.** At 360×640 and in landscape there is no horizontal scroll. With `&lang=he` the page is `dir=rtl` and the bar is still LEFT on the left.

On-device checks stay manual: iOS Safari and Android Chrome, the permission prompt, wake lock and the vibration.

---

## 8. Open points for the orchestrator

1. **The grip gate rarely fires.** `predict_stream` checks only the rest before the stream's first movement. The page starts the stream right after a tap on its own screen, so that rest almost always has the screen facing the user. A screen-away touch later in the session is scored as is. **Proposal:** add the per-attempt rule (`final_robust.rest_up_z` of each attempt below 0 → not counted, plus the grip message) to Python first, then port it. Until then the app has only the non-blocking `screenAwayHint`.
2. **Stopping thresholds belong to m4.** The thresholds and the detector are lane m4's; for the CNN they are not refitted (lane prod). The footer and the info sheet say nothing about this; the code comments and the report do.
3. **New export formats.** `m4_production.json`, `m1_phys6.json` and `ref_yaw8.json` do not exist yet. The `DetectorBundle` shape in `types.ts` is a proposal for the Python exporter.
4. **Demo realism.** Whether simulated touches pass the segment classifier is untested (§3).
5. **Offline use.** It depends on an earlier collector visit (§2.2). Changing `sw.js` to pre-cache the predict chunk is possible but out of scope.
