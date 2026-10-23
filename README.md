# NoseTouch

A person holds a smartphone with the arm fully extended in front of them and brings it in to touch their nose. Using only the phone's motion sensors, can we tell whether they used the left or the right hand? NoseTouch answers this in three stages: (1) a data-collection platform, (2) model building, (3) a predictor the PI can run on their own phone. This repository holds Stage 1. It contains a web app (PWA) that anyone can open on an iPhone or an Android phone without installing anything. The app records labelled trials with both hands, 10 phone orientations, 3 speeds, 3 start positions and 9 deliberately odd movement variants. The repository also has a free upload backend, a simulator that plays synthetic participants, and a Python starter kit for Stage 2. Start with the [project overview](docs/00_PROJECT_OVERVIEW.md).

- **Status:** Stage 1. App 0.1.0, data schema 1.1.0, protocol 1.0.0. It has been tested with unit tests and the simulator but not yet on real phones or with participants. See [Status and limitations](#status-and-limitations).
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
  tests/, e2e/              Vitest unit tests, Playwright end-to-end test
  public/config.json        Study configuration read at runtime (upload endpoint, consent version, ...)
backend/                    Upload backends; google-apps-script/ is the default (Drive files + index Sheet)
server/                     Local receiver (Python standard library only) for lab sessions and development
analysis/                   Python starter kit: the motionlab package, scripts/, tests/
schema/records.schema.json  The data contract (JSON Schema 2020-12)
docs/                       Project documentation, 00 to 08
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

Open <http://localhost:5173/?sim=1>. A built-in simulator plays the participant: it holds still, reacts to the GO beep, moves, and holds the phone at the nose. That lets the whole flow run without a phone: consent, profile, sensor check, calibration, a session, review and export. The app never remembers simulation mode between runs, and every record it produces carries `simulated: true`. Add `&lang=he` for Hebrew. Without `?sim=1`, a laptop has no motion sensors, so the sensor check cannot measure a motion rate.

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

By default the app uploads nothing (`"mode": "none"` in `app/public/config.json`). Records stay on the phone and can be exported from the Data screen. To collect data centrally:

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
npm run synth -- --participants 12 --preset standard --out ../analysis/data/synthetic
```

`synth` drives the real recording pipeline (protocol, simulated participant, `TrialController`, `TrialRecord`) and writes the same folder layout as the backend. It also accepts `--seed`, `--left-fraction` and `--validate`; see the header of `app/scripts/synth.ts`. Then:

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
npm run e2e                       # Playwright: a full participant journey in simulation mode
cd ..
python -m pytest analysis/tests server -q   # with the analysis venv active
```

CI (`.github/workflows/ci.yml`) runs the typecheck, the unit tests and pytest on pushes to `main` and on pull requests. The deploy workflow also runs `npm test` before it publishes.

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
| [06 Considerations](docs/06_CONSIDERATIONS.md) | Prioritised checklist: ethics and law, privacy, safety, confounds, statistics, recruitment |
| [07 Deployment](docs/07_DEPLOYMENT.md) | Local run, phone tunnel, GitHub Pages, backend configuration, troubleshooting |
| [08 Stage 3 predictor](docs/08_STAGE3_PREDICTOR.md) | Predict mode, model packaging, the PI's evaluation protocol |
| [backend/README.md](backend/README.md), [Apps Script setup](backend/google-apps-script/README.md) | Backend options; step-by-step Apps Script deployment |
| [server/README.md](server/README.md) | Local receiver |
| [analysis/README.md](analysis/README.md) | Python kit: setup, data layout, QA, baselines, adding a model |

## Status and limitations

**Implemented in 0.1.0:**

- The collector PWA: consent with an 18+ gate, a profile with the Edinburgh short form, a sensor check, three-step calibration, the `quick`, `standard`, `full` and `custom` presets, and a trial runner with review, redo and problem flags. It also has data export and deletion, researcher settings and a live sandbox.
- Offline storage with an upload queue.
- The Apps Script backend and the local receiver.
- The simulator and the synthetic generator.
- The Python starter kit with baseline models.
- CI, GitHub Pages deployment, and this documentation.

**Not done yet:**

- **Real-device verification.** So far everything has been tested only with unit tests and the simulator. The sign conventions, sample rates and touch behaviour must be confirmed on at least one iPhone and one Android phone (checklist in [03](docs/03_SENSORS_AND_PLATFORMS.md)).
- **Ethics approval and participant data.** Do not collect data from anyone but yourself before approval ([Considerations](docs/06_CONSIDERATIONS.md)).
- **The phyphox experiment file** for the high-rate validation cohort.
- **Stages 2 and 3.** Stage 2 has only the starter baselines. Stage 3 has only the `Predictor` interface in `app/src/core/predictor.ts`.

**Known limitations:**

- **Sample rate.** Web pages get at most about 60 Hz of motion data, and Chrome quantises the values. Higher rates need native apps, and iOS caps even native apps at 100 Hz ([Apple, WWDC23](https://developer.apple.com/videos/play/wwdc2023/10179)). The project uses phyphox as the high-rate reference instead of building a native app.
- **Foreground only.** Recording works only in the foreground with the screen on. If the page is hidden during a trial, the trial is marked `interrupted`.
- **Safari storage.** Safari deletes a site's stored data after 7 days without use; Home Screen web apps are exempt ([WebKit](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)). The app therefore uploads after every record and offers an export.
- **Screen rotation on iOS.** iOS web pages cannot lock the screen orientation, so the display may rotate in landscape and upside-down trials. The data are unaffected, because the sensor frame is fixed to the device.
- **Other browsers.**
  - Firefox for Android works but has been tested less.
  - Brave for Android has been reported to block motion sensors ([2021 report](https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620)).
  - The app detects in-app browsers (Instagram, Facebook, ...) and asks the participant to open the link in Safari or Chrome.
- **Public endpoint.** The upload endpoint is public by design. The backend validates records and caps their size. It never stores a rejected upload: it logs only metadata and identifiers in a `quarantine` sheet, at most 500 rows per day. Keep the Drive folder private.
- **Deploy branch.** The deploy workflow listens for pushes to `main`. If your default branch is `master`, rename it or edit the workflow.
