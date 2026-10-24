# NoseTouch

A person holds a smartphone with the arm fully extended in front of them and brings it in to touch their nose. Using only the phone's motion sensors, can we tell whether they used the left or the right hand? NoseTouch answers this in three stages: (1) a data-collection platform, (2) model building, (3) a predictor the PI can run on their own phone. This repository holds Stage 1. It contains a web app (PWA) that anyone can open on an iPhone or an Android phone without installing anything. Version 0.2.0 runs a hands-free visit of about 5 minutes (the `street` protocol): welcome and consent on one screen, a short profile, a How-to card with an animation, one free trial whose hand the participant reports, then blocks that chain the trials without taps — the most important conditions first, so stopping early loses little. It records labelled trials with both hands, upright and sideways grips with the screen facing the participant (never facing away: the analysis synthesises those), 3 speeds, 3 start positions and deliberately varied movements (each visit draws four of them, always including a tilted touch), plus every motion signal the browser offers and the return of the arm. The repository also has upload backends, a simulator that plays synthetic participants, and a Python starter kit for Stage 2. Start with the [project overview](docs/00_PROJECT_OVERVIEW.md).

- **Status:** Stage 1. App 0.2.0, data schema 1.2.0, protocol 2.0.0. Version 0.1.0 was piloted once on the researcher's Android phone; 0.2.0 has been tested with unit tests and with end-to-end journeys of the simulated participant, not yet on real phones. See [Status and limitations](#status-and-limitations).
- **Runs on:** any iOS browser (they all use WebKit), and Chrome or Samsung Internet on Android. The interface is in English and Hebrew.
- **Data:** raw sensor streams at about 60 Hz, one JSON record per trial attempt, defined by [schema/records.schema.json](schema/records.schema.json).

## Repository map

```text
app/                        Collector PWA (Vite + TypeScript + Preact)
  src/core/                 DOM-free logic: protocol, trial controller, online_v1 segmentation, calibration,
                            sensors, storage, uploader, exporter, mirror, Predictor interface
  src/sim/                  Kinematic simulator (minimum-jerk model) used by ?sim=1, the tests and synth
  src/app/, src/ui/         Screens and UI components
  src/i18n/                 English and Hebrew strings; en/common.ts holds the exact participant instructions
  scripts/synth.ts          Synthetic dataset generator (runs the real recording pipeline)
  tests/, e2e/              Vitest unit tests; Playwright journeys of the simulated participant (headless, silent)
  public/config.json        Study configuration read at runtime (upload endpoint, consent version, ...)
backend/                    Upload backends: heroku/ (live uploads into Postgres, admin page) and
                            google-apps-script/ (Drive files + index Sheet)
server/                     Local receiver (Python standard library only) for lab sessions and development
analysis/                   Python starter kit: the motionlab package, scripts/, tests/
schema/records.schema.json  The data contract (JSON Schema 2020-12)
docs/                       Project documentation, 00 to 10
.github/workflows/          CI (ci.yml) and GitHub Pages deployment (deploy.yml)
```

## Quick starts

You need Node.js 24 (the version CI uses) and Python 3.10 or newer for the analysis kit (CI uses 3.11).

### (a) Try it on a laptop with a simulated participant

```bash
cd app
npm install
npm run dev
```

Open <http://localhost:5173/?sim=1>. A built-in simulator plays the participant: it holds still with the arm out, reacts to GO, brings the phone to the nose, holds it there and stretches the arm out again. That lets the whole visit run without a phone: welcome and consent, the profile, the How-to card, the free trial and its hand question, the hands-free blocks, the checkpoint, done and export (about 3 minutes). In simulation mode the app never beeps or speaks, never remembers simulation mode between runs, and marks every record `simulated: true`. Add `&lang=he` for Hebrew. Researchers: add `&setup=1` to pick another preset or a short custom session after the profile (also in Settings). Without `?sim=1`, a laptop has no motion sensors, so after the profile the app opens its sensor repair screen.

### (b) Run it on a phone

Motion sensors work only on HTTPS pages. On a plain `http://192.168.x.x` address, iOS answers the permission request with "denied" and never shows a prompt ([WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/DeviceMotionEvent.cpp)).

- **Deployed build (use this for participants).** One-time setup: in the repository on GitHub, set Settings > Pages > Source to "GitHub Actions". After that, every push to `main` that changes `app/`, `schema/` or the workflow file runs `.github/workflows/deploy.yml` (or start it by hand under Actions), which tests, builds and publishes the app at `https://<github-user>.github.io/<repo>/`. The app uses a relative base path, so any repository name works. Share the URL or a QR code of it. Details are in [Deployment](docs/07_DEPLOYMENT.md).
- **Your dev server, through a tunnel.** A Cloudflare quick tunnel gives the dev server a real certificate and needs no account ([Cloudflare docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)):

  ```bash
  # terminal 1
  cd app
  npm run dev -- --host
  # terminal 2 (install once: winget install --id Cloudflare.cloudflared, or: brew install cloudflared)
  cloudflared tunnel --url http://localhost:5173
  ```

  Open the printed `https://<random-words>.trycloudflare.com` address on the phone. The hostname changes on every run. `vite.config.ts` already allows `.trycloudflare.com` hosts. Vite 6.0.9 and later block hosts that are not on that list ([advisory](https://github.com/vitejs/vite/security/advisories/GHSA-vg6x-rcgg-rjx6)). Avoid self-signed certificates; `npm run dev:phone` is only a fallback. Chrome refuses to register the service worker on a self-signed page ([Chromium source](https://github.com/endlessm/chromium-browser/blob/aa8c819d5ad2fcb3854a688a0401975eca721f43/content/browser/service_worker/service_worker_consts.cc)), and iOS behaviour after the certificate warning is undocumented.

### (c) Collect data centrally

Without an upload configuration the app uploads nothing (`"mode": "none"`, the built-in default): records stay on the phone and can be exported from the Data screen. The v2 study uploads live to [backend/heroku](backend/heroku/README.md), a small FastAPI service with Postgres on Heroku and an admin page, which the app reaches with upload mode `json_post` (`"endpoint": "https://<your-app>.herokuapp.com/api/ingest"`, set in this repository's `app/public/config.json`); its README covers deployment, the admin page and the export for the analysis kit (`analysis/scripts/pull_live.py`). Every saved trial and session update is sent at once, so a participant who stops early loses nothing. Without a server of your own, the free Google Apps Script backend works too:

1. Set up the free Google Apps Script backend by following [backend/google-apps-script/README.md](backend/google-apps-script/README.md). In short, create the script from a dedicated Gmail account, paste `Code.gs` and `appsscript.json`, run `setup()`, and deploy it as a web app with "Execute as: Me" and "Who has access: Anyone".
2. Point the app at the deployment in `app/public/config.json`:

   ```json
   "upload": {
     "mode": "apps_script",
     "endpoint": "https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec",
     "study_key": "<same value as the STUDY_KEY script property>"
   }
   ```

3. The app reads `config.json` at runtime, so no code change is needed. On GitHub Pages, commit the file and the workflow redeploys.

The study key only filters spam. It is not a secret, because it ships in the public page. If you change it, phones pick up the new key only when the app is reloaded; until then the backend rejects their uploads without storing them, and the app keeps those records for a re-send with "Upload now" ([Deployment §4.4](docs/07_DEPLOYMENT.md#44-edit-apppublicconfigjson)). To update the backend code and keep the same URL, use Deploy > Manage deployments > Edit > Version: New version. A new deployment would change the URL ([Google docs](https://developers.google.com/apps-script/concepts/deployments)). For lab sessions or development, `server/receiver.py` stores the same folder layout on a laptop: run `python server/receiver.py --port 8787` and set upload mode `json_post` with endpoint `/api/ingest` (served through the Vite proxy). See [server/README.md](server/README.md). Backend options and their trade-offs are in [backend/README.md](backend/README.md).

### (d) Generate synthetic data and run the analysis kit

```bash
cd app
npm run synth -- --participants 12 --preset street --out ../analysis/data/synthetic --validate
```

`synth` drives the real recording pipeline (protocol, simulated participant, `TrialController`, `TrialRecord`) and writes the same folder layout as the backend. `street` is the default preset and simulates whole v2 visits (implicit calibration, the reported hand of the choice trial, auto-accept with re-runs, the optional bonus). It also accepts `--seed`, `--left-fraction` and more; see the header of `app/scripts/synth.ts`. Then:

```bash
cd ../analysis
py -3.11 -m venv .venv           # macOS/Linux: python3 -m venv .venv
.venv\Scripts\Activate.ps1       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/qa_report.py data/synthetic        # dataset QA
python scripts/train_baseline.py data/synthetic   # subject-wise CV of the baseline models
```

Reports (Markdown and PNG) go to `analysis/reports/<name>/`. Each script lists its options with `--help`. Git ignores `analysis/data/` and `analysis/reports/`, so participant data never ends up in the repository. The kit is documented in [analysis/README.md](analysis/README.md).

### (e) Run the tests

```bash
cd app
npm run typecheck
npm test                          # Vitest unit tests (Node, no browser)
npx playwright install chromium   # once
npm run e2e                       # Playwright, headless: the simulated participant's journeys (about 5 minutes)
cd ..
python -m pytest analysis/tests server -q   # with the analysis venv active
```

`npm run e2e` covers a whole street visit (schema-checked export, timings printed), finishing early, the setup screen, the researchers' tap-review mode, live uploads (answered by the test itself), a whole visit in Hebrew, back and pause, resuming, double taps, small phone screens and sideways blocks in landscape. It never makes a sound: every spec runs through `e2e/fixtures.ts`, which replaces speech and Web Audio in the page, and Chromium starts with `--mute-audio`. Nothing leaves the computer either: the same fixtures serve `config.json` with uploads off and block (and fail the test on) every request to another host than the dev server. Never run it headed. `npm run e2e:live` is different: it runs the simulated journeys against the deployed site, and their records **are uploaded** to the live backend (marked `simulated: true`); run it only on purpose ([Deployment](docs/07_DEPLOYMENT.md)). `SCREENS=1 npx playwright test e2e/screens.spec.ts` saves a screenshot of every screen. The Heroku upload service has its own tests (`backend/heroku`, see its README).

CI (`.github/workflows/ci.yml`) runs the typecheck, the unit tests and pytest on pushes to `main` and on pull requests; it does not run the e2e journeys (`app/tests/e2eHarness.test.ts` checks their silence set-up instead). The deploy workflow also runs `npm test` before it publishes.

## Read this before touching the data

These four conventions decide whether left and right come out the right way round. Details and more sources are in [Sensors and platforms](docs/03_SENSORS_AND_PLATFORMS.md).

1. `rr_alpha`, `rr_beta` and `rr_gamma` are the rotation rates about the device **x, y and z** axes, in deg/s ([W3C Candidate Recommendation Draft, 12 Feb 2025](https://w3c.github.io/deviceorientation/)). [MDN's page](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMotionEventRotationRate/alpha) still says alpha is about z, which is outdated.
2. Every iOS browser runs WebKit, which reports `acc_*` and `accg_*` with all three axes inverted. A face-up phone reads `accg_z` ≈ −9.81 instead of +9.81 ([WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm)). The rotation rate is not inverted. The phone flips nothing; analysis uses the session's measured `calibration.gravity_sign`.
3. Browsers deliver about 60 Hz. Chrome rounds values to 0.1 m/s² and 0.1 deg/s; iOS does not round ([Chromium source](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). Use the per-sample times `t`, never `event.interval`.
4. The `orientation` stream (Euler angles) is metadata only. It hits gimbal lock in the upright start pose, where beta ≈ 90° ([W3C](https://w3c.github.io/deviceorientation/)).

## Documentation

| Document | What it covers |
|---|---|
| [00 Project overview](docs/00_PROJECT_OVERVIEW.md) | Problem, the 3 stages with acceptance criteria, architecture, data flow, plan, risks, glossary |
| [01 Protocol](docs/01_PROTOCOL.md) | Factors and exact instructions, presets, trial timeline, label quality, recruitment, phyphox validation sub-study |
| [02 Data schema](docs/02_DATA_SCHEMA.md) | Record types and fields, units, storage layout, loading in Python, exclusion rules |
| [03 Sensors and platforms](docs/03_SENSORS_AND_PLATFORMS.md) | iOS vs Android behaviour, calibration, normalisation rules, high-rate alternatives, real-device checklist |
| [04 Segmentation](docs/04_SEGMENTATION.md) | `online_v1`, its parameters and overrides, offline methods, validation |
| [05 Modeling plan](docs/05_MODELING_PLAN.md) | Preprocessing, mirror symmetry, features, baselines, evaluation, subgroup analysis, handedness add-on |
| [06 Considerations](docs/06_CONSIDERATIONS.md) | Prioritised checklist: consent and law, privacy, safety, confounds, statistics, recruitment |
| [07 Deployment](docs/07_DEPLOYMENT.md) | Local run, phone tunnel, GitHub Pages, backend configuration, troubleshooting |
| [08 Stage 3 predictor](docs/08_STAGE3_PREDICTOR.md) | Predict mode, model packaging, the PI's evaluation protocol |
| [09 Stage 2 and 3 plan](docs/09_STAGE2_3_PLAN.md) | Decisions for training, the predictor link, the report and the deck; milestones |
| [10 Field guide](docs/10_FIELD_GUIDE.md) | One page for running sessions: the link, what to say, browsers, permission fixes, live uploads, pulling and deleting data |
| [backend/README.md](backend/README.md), [Apps Script setup](backend/google-apps-script/README.md) | Backend options; step-by-step Apps Script deployment |
| [server/README.md](server/README.md) | Local receiver |
| [analysis/README.md](analysis/README.md) | Python kit: setup, data layout, QA, baselines, adding a model |

## Status and limitations

**Implemented in 0.2.0 (the `street` visit, protocol 2.0.0):**

- The participant path: welcome and consent on one screen (18+ and no arm or hand injury in one checkbox), a short profile (name, age, sex, dominant hand, usual phone hand, optional height), then the session. The sensor check opens only to repair a phone that cannot record; calibration is implicit (taken from the first still upright trial), so no table and no tilt step.
- The hands-free session: How-to card with an animation, one free trial with the hand question, blocks of tiers 1–3 (upright, sideways, varied movements) whose trials chain without taps, a checkpoint with an optional bonus, a pause sheet (hold-to-pause, back button, redo, wrong hand, finish now) and a done screen with the live upload state and the participant code.
- Every extra signal the browser offers: absolute orientation and its quaternion (Chrome Android), compass accuracy (iOS), magnetometer where exposed, screen rotations, touch size and force, and the return of the arm after each touch.
- Live uploads after every record (`json_post` to the Heroku service, or Apps Script), names uploaded with the participants' consent.
- For researchers: the setup screen with the `street`, `quick`, `standard`, `full` and `custom` presets (`?setup=1` or Settings), a tap-review mode for lab sessions, data export and deletion, settings and a live sandbox.
- The simulator and the synthetic generator (whole street visits), the Python starter kit (schema 1.2.0, implicit calibration, screen-away synthesis), CI, GitHub Pages deployment and this documentation.

**Not done yet:**

- **Real-device verification of 0.2.0.** The hands-free chain, the implicit calibration and the new sensor streams have run only in the simulator. Check on at least one iPhone and one Android phone: the motion prompt on the Start tap, beeps after the block-intro tap (iOS plays audio only after a tap and stops it when the page is hidden), the screen staying on through a block (wake lock), sideways grips that rotate the page, and the sign conventions (checklist in [03](docs/03_SENSORS_AND_PLATFORMS.md)).
- **Study details in `app/public/config.json`:** the upload endpoint and study key are set (the Heroku service); the researcher's contact is still empty, so the consent tells participants to contact the researcher who sent them the link. The Hebrew consent text has not yet been proofread by a native speaker.
- **The phyphox experiment file** for the high-rate validation cohort.
- **Stages 2 and 3.** Stage 2 has only the starter baselines; the plan is [09](docs/09_STAGE2_3_PLAN.md). Stage 3 has only the `Predictor` interface in `app/src/core/predictor.ts`.

**Known limitations:**

- **Sample rate.** Web pages get at most about 60 Hz of motion data, and Chrome quantises the values. Higher rates need native apps, and iOS caps even native apps at 100 Hz ([Apple, WWDC23](https://developer.apple.com/videos/play/wwdc2023/10179)). The project uses phyphox as the high-rate reference instead of building a native app.
- **Foreground only.** Recording works only in the foreground with the screen on. If the page is hidden during a trial, the trial stops and the session waits on its pause sheet; an attempt that had reached GO is kept as `interrupted`, an earlier one is discarded, and one that was already recording the return of the arm stays complete (warnings `return_cut_short` and `hidden_after_end`; `quality.hidden_during` covers only the movement).
- **Safari storage.** Safari deletes a site's stored data after 7 days without use; Home Screen web apps are exempt ([WebKit](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)). The app therefore uploads after every record and offers an export.
- **Screen rotation on iOS.** iOS web pages cannot lock the screen orientation, so the display may rotate in sideways and upside-down trials. The data are unaffected, because the sensor frame is fixed to the device, and the rotations are recorded (`streams.screen_events`; one during a movement adds the warning `screen_rotated_during_movement`).
- **Other browsers.**
  - Firefox for Android works but has been tested less.
  - Brave for Android has been reported to block motion sensors ([2021 report](https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620)).
  - The app detects in-app browsers (WhatsApp, Instagram, Facebook, ...) and asks the participant to open the link in Safari or Chrome, with a "Copy the link" button.
- **Public endpoint.** The upload endpoint is public by design. The backends validate records and cap their size. They never store a rejected upload: they log only metadata and identifiers (Heroku: `ingest_log`; Apps Script: a `quarantine` sheet), at most 500 refusals per day. Keep the database, the admin token and the Drive folder private: the records hold participants' names.
- **Deploy branch.** The deploy workflow listens for pushes to `main`. If your default branch is `master`, rename it or edit the workflow.
