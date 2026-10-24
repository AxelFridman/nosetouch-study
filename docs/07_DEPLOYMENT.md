# 07 · Deployment and operations

This guide takes the study from the repository to a live link and back to data on your laptop, in the order you will do it. Commands are for Windows PowerShell, the student's machine. Differences for macOS and Linux are noted where they matter.

**Related:** [Project overview](00_PROJECT_OVERVIEW.md) · [Data schema](02_DATA_SCHEMA.md) · [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md) · [Considerations](06_CONSIDERATIONS.md) · [Stage 3 predictor](08_STAGE3_PREDICTOR.md) · [Field guide](10_FIELD_GUIDE.md) · upload service [`backend/heroku/`](../backend/heroku/README.md) · alternative backend [`backend/google-apps-script/`](../backend/google-apps-script/) · receiver [`server/receiver.py`](../server/receiver.py)

## Overview

| Piece | Where it runs | Source | How you change it |
|---|---|---|---|
| Collector app (PWA) | Participants' phone browsers. GitHub Pages serves it as static files, at <https://axelfridman.github.io/nosetouch-study/> for this study. | `app/` | Push to `main`. The deploy workflow rebuilds it. |
| Study settings | Fetched by the app every time it starts; the last copy is kept for offline starts | `app/public/config.json` | Edit the file and push. No code change. |
| Upload service (this study) | A FastAPI service with Postgres on Heroku: live uploads, an admin page, exports, deletion | `backend/heroku/` | `backend/heroku/deploy.sh <app>`; see its [README](../backend/heroku/README.md). |
| Upload backend (alternative) | Google Apps Script web app in a study Google account. Writes to Drive and to an index Sheet. | `backend/google-apps-script/` | Manage deployments > New version. The URL stays the same. |
| Local receiver (optional) | Lab laptop | `server/receiver.py` | Run it during a lab session. |
| Analysis kit | Your laptop | `analysis/` | Python |

Data path: phone (IndexedDB outbox) → one HTTPS POST per record, sent right after every save → the Heroku service → Postgres → `analysis/scripts/pull_live.py` → `analysis/data/live/` (git-ignored), in the folder layout `<study_code>/<participant_id>/<session_id>/*.json`. Apps Script writes the same layout to Drive, and the receiver and the synthetic generator write it to disk ([Data schema §7](02_DATA_SCHEMA.md#7-storage-layout)).

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Node.js and npm | 24 (CI uses 24) | Check with `node --version`. |
| Git and a GitHub account | any | On the Free plan the repository must be public (§3). |
| Python | 3.10 or newer (CI uses 3.11) | Windows: the `py` launcher |
| cloudflared | current | See §2. |
| Heroku account and CLI | for the upload service | [backend/heroku/README.md](../backend/heroku/README.md); the service's URL, study key and admin token live in `%USERPROFILE%\.nosetouch\heroku.env`, never in the repository. |
| Google account | only for the Apps Script alternative | See §4. |
| Test phones | one iPhone and one Android phone | Borrow if needed. |

## 1. Run locally with the simulator

```powershell
cd app
npm ci
npm run dev
```

Open <http://localhost:5173/?sim=1>. The built-in simulator plays a participant. It holds still, reacts to the GO cue, performs the instructed movement with the instructed hand and stretches the arm out again, so the whole street visit runs in a desktop browser: welcome and consent, profile, the How-to card, the free trial and its hand question, the hands-free blocks, the checkpoint, done and export. It never beeps or talks: the app silences itself in simulation mode and in automated browsers.

*Code:* records made this way carry `simulated: true`, and `?sim=1` is never remembered (`app/src/app/App.tsx`, `app/src/app/state.ts`). Reload without it to use real sensors. Without `?sim=1`, a desktop browser has no motion sensors, so the profile's Continue opens the sensor repair screen, which is expected. Add `&lang=he` for the Hebrew interface, and `&setup=1` to choose another preset or a short custom session.

| Command (in `app/`) | What it does |
|---|---|
| `npm run typecheck` | Checks the TypeScript types. |
| `npm test` | Runs the unit tests (Vitest). |
| `npm run e2e` | Playwright drives the participant journeys in simulation mode, headless and silent (`e2e/fixtures.ts`, `--mute-audio`), in a Pixel 7 viewport, and starts the dev server itself (about 5 minutes). Nothing leaves the computer: the fixtures serve `config.json` with uploads off and block, and fail the test on, every request to another host. The first time, run `npx playwright install chromium`. |
| `npm run e2e:live` | Smoke test of the deployed site (default: the GitHub Pages URL; `LIVE_URL=…` for another): the simulated journeys of `flow.spec.ts` and the boot check against the production build. Its records **are uploaded** to the backend in the deployed `config.json` (marked `simulated: true`, dropped by rule S1 and by `pull_live.py --real-only`). Run it only on purpose. |
| `npm run build` | Typechecks, then builds for production into `app/dist/`. |
| `npm run preview` | Serves `app/dist/` on port 4173 on all network interfaces, with the service worker. |
| `npm run synth -- …` | Generates a synthetic dataset through the real recording pipeline (see below). |

Synthetic dataset, in the same folder layout as the backend:

```powershell
npm run synth -- --participants 12 --preset street --seed 42 --left-fraction 0.3 --out ../analysis/data/synthetic
```

Optional flags are `--study synthetic-v1` and `--failure-rate 0.03` (`app/scripts/synth.ts`). Use the dataset to exercise the analysis kit (§7) before real data exist.

## 2. Test on real phones through a quick tunnel

Phones need HTTPS. Motion sensors require a secure context, and on a plain `http://192.168.x.x` page iOS answers the motion-permission request with "denied" without showing a prompt ([WebKit][wk-permission]). A Cloudflare quick tunnel gives the laptop's dev server a public HTTPS address with a valid certificate. It needs no account ([Cloudflare][trycloudflare]).

Install it once:

```powershell
winget install --id Cloudflare.cloudflared
```

On macOS run `brew install cloudflared`. For other systems see the [downloads page][cloudflared-downloads]. Open a new terminal afterwards so that `cloudflared` is on the PATH.

Run these in two terminals:

```powershell
# terminal 1
cd app
npm run dev -- --host

# terminal 2
cloudflared tunnel --url http://localhost:5173
```

cloudflared prints an address like `https://<random-words>.trycloudflare.com`. Open it on the phone: send it to yourself or turn it into a QR code. Then work through the real-device checklist in [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md).

Things to know:

- **Allowed hosts.** *Code:* `app/vite.config.ts` already allows `.trycloudflare.com` hosts. Vite 6.0.9 and later reject requests whose `Host` header is not on that list ([advisory GHSA-vg6x-rcgg-rjx6][vite-cve]). The message "Blocked request. This host (…) is not allowed" means you used a different tunnel provider. Add its domain to `tunnelHosts`, and never set `allowedHosts: true` ([Vite server options][vite-server-options]).
- **New hostname, new origin.** The hostname changes on every run. A new hostname is a new origin, so the phone asks for permission again and starts with empty local storage. Upload or export what you recorded before you stop a tunnel.
- **No service worker on the dev server.** *Code:* `app/src/main.tsx` registers `sw.js` only in production builds. To test offline use and Add to Home Screen, tunnel the production build instead: run `npm run build`, then `npm run preview`, then `cloudflared tunnel --url http://localhost:4173`. Vite's preview server reuses `server.proxy` (the `/api` proxy), and `vite.config.ts` sets `preview.allowedHosts` as well.
- **Quick tunnels are for testing.** They have no uptime guarantee and allow at most 200 requests in flight ([Cloudflare][trycloudflare]). The dev server loads many small module files per page, so use the preview server when several phones connect at once.
- **Hot reload.** If hot reload does not reach the phone, reload by hand. Setting `server.hmr.clientPort: 443` may fix it, but this is unverified.

**Why not a self-signed certificate.** `npm run dev:phone` serves HTTPS on the LAN with a self-signed certificate (`@vitejs/plugin-basic-ssl`). Chrome refuses to register a service worker when the script was fetched with a certificate error ([Chromium source][chromium-sw-ssl]). How iOS handles the motion permission after the user clicks through a certificate warning is undocumented. So you cannot test offline behaviour or installation this way, and a failure may come from the certificate rather than the app. Keep `dev:phone` as a last resort for a lab without internet access. mkcert also works if its root CA is fully trusted on the phone, but it takes more setup ([mkcert][mkcert]).

## 3. Deploy to GitHub Pages

One-time setup:

1. Push the repository to GitHub. On GitHub Free the repository must be public for Pages. GitHub Pro, free with the Student Developer Pack, allows a private repository, but the site itself is always public ([creating a Pages site][gh-pages-create]; [plans][gh-plans]; [Student Pack][student-pack]).
2. Go to Settings > Pages > Build and deployment and set Source to **GitHub Actions**.
3. Push to `main`, or start the workflow by hand: Actions > "Deploy app to GitHub Pages" > Run workflow.

What `.github/workflows/deploy.yml` does:

| Step | Detail |
|---|---|
| Trigger | A push to `main` that changes `app/**`, `schema/**` or the workflow file, or a manual run. |
| Build job | Uses Node 24 and runs `npm ci`, `npm test` and `npm run build` in `app/`. A failing test or a type error stops the deploy, and the live site stays as it was. |
| Publish | Uploads `app/dist` as the Pages artifact. `actions/deploy-pages` then publishes it to the `github-pages` environment. |
| Concurrency | One deployment runs at a time. A newer push cancels an older run. |

The site appears at `https://<user>.github.io/<repo>/`. The URL is shown in the run summary and under Settings > Pages. *Code:* `base: './'` in `app/vite.config.ts` makes every asset URL relative, so the build works under any repository name, sub-path or custom domain without changes. Do not change it to an absolute path.

Once participants are recording, follow these release rules:

1. Every push to `main` that touches `app/` redeploys the app participants use, and that includes edits to `config.json`. Develop on a branch and merge to `main` only to release.
2. Bump the version in `app/package.json` and `APP_VERSION` in `app/src/core/constants.ts` together. A unit test checks that they match, and every record carries `app_version`.
3. Tag each release used for collection: run `git tag collect-0.2.0`, then `git push --tags`.
4. Do not deploy on a collection day or on the day of the PI's evaluation.
5. *Code:* `.github/workflows/ci.yml` runs the typecheck, the unit tests and the Python tests (`analysis/tests`, `server`) on pushes to `main` and on pull requests. Pushes to other branches are not tested until you open a pull request.

Limits: the published site can be at most 1 GB, and bandwidth has a soft limit of 100 GB per month ([limits][gh-pages-limits]). GitHub records visitors' IP addresses, and Pages is not meant for sensitive transactions ([about Pages][gh-pages-about]). Mention the IP logging in the consent text ([Considerations §1](06_CONSIDERATIONS.md#1-ethics-and-law)).

## 4. Connect the backend and edit config.json

**This study uploads to the Heroku service** (`upload.mode: "json_post"`). Its README covers deployment, the admin page, the export for the analysis kit and deletion: [backend/heroku/README.md](../backend/heroku/README.md); the field steps are in [10](10_FIELD_GUIDE.md). §4.1–4.3 describe the Google Apps Script backend, the free alternative that needs no server of your own; §4.4 applies to both.

### 4.1 What the Apps Script backend does

- **What it is.** A free Google Apps Script web app that runs in the study's own Google account. There is no server and no billing.
- **How the app sends.** *Code:* the app sends each record as one `UploadEnvelope` (`{kind, study_key, sent_at, record}`), in a `POST` with a plain string body and no custom headers (`app/src/core/uploader.ts`). That makes it a CORS "simple request", so the browser sends no preflight, which Apps Script could not answer ([Fetch standard][fetch-spec]).
- **How the reply is read.** Apps Script runs `doPost` and then redirects to `script.googleusercontent.com`. That JSON response carries `Access-Control-Allow-Origin: *`, so the app can read `{ok: true}` and mark the record as sent ([Apps Script content service][gas-content]; [Tanaike's tests][tanaike-webapps]).
- **Storage.** The script stores each record as one JSON file in a private Drive folder, in the layout of [Data schema §7](02_DATA_SCHEMA.md#7-storage-layout), and appends one index row per record to a Google Sheet. Raw arrays never go into the Sheet, because a cell holds at most 50,000 characters ([Google][drive-limits]).
- **Retries are safe.** Re-sending a record overwrites its own file.
- **More detail.** The code and the detailed setup are in [`backend/google-apps-script/`](../backend/google-apps-script/) and its README. The security requirements (schema validation, size cap, no storage of rejected uploads) are in [Considerations §2](06_CONSIDERATIONS.md#2-privacy-and-security).

### 4.2 Deploy the web app

1. Create a dedicated Google account for the study (a consumer `@gmail.com` account) and turn on 2-step verification. University Google Workspace domains may not allow anonymous access to web apps. This is medium confidence, from a Google Developer Expert ([labnol][labnol]).
2. Signed in to that account, open <https://script.google.com> and create a project. Add the files from `backend/google-apps-script/`, then set up the Drive folder, the Sheet and the study key as its README describes.
3. Choose Deploy > New deployment, click the gear icon (Select type) and pick **Web app**. Set **Execute as: Me** and **Who has access: Anyone**. Deploy, authorise, and tick every requested scope: since January 2025 Google asks for each scope separately, and an unticked scope breaks the script ([release notes][gas-releases]). Copy the URL that ends in `/exec` ([web apps guide][gas-web]).
   - "Anyone" means anonymous access. "Anyone with Google account" forces a sign-in that a cross-origin `fetch` cannot complete, so every upload fails ([web app manifest][gas-manifest-webapp]).
4. Smoke-test it. Open the `/exec` URL in a browser: the health route should answer with JSON that contains `"ok": true`. Then check the CORS headers. In PowerShell type `curl.exe`, because there `curl` is an alias of `Invoke-WebRequest`:

   ```powershell
   curl.exe -s -D - -o NUL -H "Origin: https://<user>.github.io" "https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec"
   ```

   Expect a `302` with `access-control-allow-origin: *` and a `location:` header that points to `script.googleusercontent.com` ([Tanaike][tanaike-webapps]). An HTML page or a sign-in redirect means the access setting is wrong.
5. Test end to end:
   1. put the URL into `config.json` (§4.4) and deploy;
   2. record one trial on a phone;
   3. check that the Done screen reports everything as sent and that the file appears in Drive.

   Errors show up in the Apps Script editor under **Executions**.
6. Before the first lab session, load-test with about 30 parallel uploads. *Fact:* Apps Script allows 30 simultaneous executions per user and 6 minutes per execution. With "Execute as: Me", every participant's request counts against the owner ([quotas][gas-quotas]). In practice one web app handles just under 30 simultaneous connections ([Tanaike][tanaike-webapps]). *Code:* each phone keeps one request in flight and backs off after failures.
7. The account's 15 GB is shared with Gmail and Photos ([Google One][googleone-15gb]). The study needs far less ([Considerations §7](06_CONSIDERATIONS.md#7-data-management)), but do not use the account for anything else.

### 4.3 Update the script without changing the URL

To keep the URL:

1. Save the code.
2. Choose Deploy > Manage deployments and select the active deployment.
3. Click Edit (the pencil icon), set Version to **New version** and click Deploy.

The `/exec` URL stays the same. **New deployment** creates a new URL, and `config.json` must then change. The `/dev` URL works only for the script's editors, so never give it to participants ([deployments][gas-deployments]). After you add a new Google service to the script, run any function once in the editor to re-authorise it. Until you do, every request fails.

### 4.4 Edit `app/public/config.json`

| Field | In this study's file | Meaning | When to change |
|---|---|---|---|
| `study_code` | `"nose-touch-v1"` | Stored in every record and used as the top folder of every export. Must match `[A-Za-z0-9_-]{1,64}`. | For a new wave, e.g. a validation cohort |
| `upload.mode` | `"json_post"` | `"json_post"` (the Heroku service, or the receiver), `"apps_script"`, or `"none"` (no upload; participants export a file). The built-in fallback is `"none"`. | Only when switching backends |
| `upload.endpoint` | `"https://nosetouch-study-14d4b1cd9830.herokuapp.com/api/ingest"` | Any `https://` URL, or a same-origin path such as `/api/ingest`. | Together with the mode |
| `upload.study_key` | set | Sent inside the body and checked by the backend (`STUDY_KEY` on Heroku). It filters spam; it is not a secret. | If junk uploads appear. Change it together with the backend's key, and see item 3 below |
| `upload_nickname` | `true` | Whether the participant's name (`nickname`) is uploaded. The consent wording follows the setting. | Names are uploaded with consent in this study ([06 §2](06_CONSIDERATIONS.md#2-privacy-and-security)); changing it changes the consent, so bump `consent_version` too |
| `consent_version` | `"2026-10-v3"` | Copied into each session. | Whenever the consent text changes |
| `contact.name`, `contact.email` | `""` | The researcher contact shown to participants; while empty, the consent says "contact the researcher who sent you this link". | Before recruiting beyond people who know you |
| `institution.en`, `institution.he` | `""` | The institution named in the consent and About texts ("Researchers at …"). Empty gives neutral wording. | If wanted |
| `region.en`, `region.he` | `""` | Where participants must be, e.g. `"Israel"` / `"ישראל"`; it adds "I'm taking part from …" to the consent checkbox. Empty omits the clause. | If recruitment is restricted (see [06](06_CONSIDERATIONS.md)) |
| `default_language` | `"en"` | `"en"` or `"he"`. Participants can switch, and `?lang=` overrides it. | For Hebrew-speaking cohorts |
| `default_preset` | `"street"` | The session every participant gets: `street`, `quick`, `standard`, `full` or `custom`. Researchers choose another one with `?setup=1`. | Per wave |
| `min_age` | `18` | The age gate. | Keep 18 |

Example for the Apps Script alternative (only the upload block differs from this study's file):

```json
{
  "study_code": "nose-touch-v1",
  "upload": {
    "mode": "apps_script",
    "endpoint": "https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec",
    "study_key": "<a random string, also set in the backend>"
  },
  "upload_nickname": true,
  "consent_version": "2026-10-v3",
  "contact": { "name": "<your name>", "email": "<your university e-mail>" },
  "institution": { "en": "", "he": "" },
  "region": { "en": "Israel", "he": "ישראל" },
  "default_language": "en",
  "default_preset": "street",
  "min_age": 18
}
```

How changes reach phones, and what can go wrong:

1. **Publishing.** The file is part of the deployed site. Edit it on GitHub (pencil icon, commit to `main`) or edit it locally and push. The deploy workflow runs because the path is under `app/`, and the change is live a few minutes later.
2. **Picked up at the next start.** *Code:* the app fetches `config.json` with `cache: 'no-store'` every time it starts (`app/src/core/config.ts`), so phones pick up the change the next time the app is opened. A phone that keeps the app open keeps the old values until it is reloaded. Offline, the app uses the copy the service worker cached (production builds precache `config.json`) or the last configuration it loaded on that phone, so an offline launch still has the study's `study_code`, `consent_version`, consent texts and upload settings. Only a phone that has never loaded the file runs on the built-in defaults. After such a fallback the app fetches the file again once the phone is back online.
3. **Changing the study key.** Set the new value in the backend's `STUDY_KEY` (a Heroku config var, or the Apps Script Script Property) and in `upload.study_key` at the same time. Phones that still have the app open send the old key until the app is reloaded. The backend rejects those uploads (`bad_study_key`) without storing them, and the app does not retry such a permanent rejection on its own. The records stay on the phone: after a reload, "Upload now" on the Done or Data screen sends them with the new key. Rotate between collection days, and ask anyone with the app open to close and reopen it.
4. **Invalid values fail silently.** *Code:* each bad value falls back to a default:
   - an unknown `upload.mode` becomes `"none"`;
   - an endpoint that is neither `https://` nor a same-origin path becomes `""`, which turns uploads off;
   - an invalid `study_code` becomes the default;
   - a file that is not valid JSON counts as a failed fetch: phones keep the last configuration they loaded, and a new phone gets the built-in defaults (uploads off).

   After every edit, open the app, record one trial and confirm that it is sent.
5. **Check the JSON.** JSON allows no comments and no trailing commas. Before committing, run `node -e "JSON.parse(require('fs').readFileSync('app/public/config.json','utf8'))"`. It prints nothing when the file is valid.
6. **Per-phone override.** On lab phones you can override the upload settings for that one device from the researcher settings screen. *Code:* this is `LocalSettings.upload_override`, stored in that browser only. Clear it after the lab session, or that phone keeps uploading to the lab target.

### 4.5 File-only mode

With `upload.mode: "none"`, nothing leaves the phone. *Code:* the Done screen then asks the participant to tap "Save / share my data" and send the file to the researcher. Chrome on Android refuses to share `.json` files, so the app retries as `….json.txt` or downloads the file instead ([Chromium share allowlist][cr-share-allowlist]; `app/src/core/exporter.ts`). Use this mode for self-testing, or when data must not leave the phone. The export is also every participant's fallback when uploads fail.

## 5. Local receiver for lab sessions

Use the receiver for supervised sessions when you want the data directly on the laptop, with no Google account involved. One example is the period while the backend or its approval is still pending.

How it fits together:

1. The phone posts to `https://<tunnel>/api/ingest`.
2. The Vite proxy forwards `/api` to `http://127.0.0.1:8787` and removes the prefix (`app/vite.config.ts`).
3. `server/receiver.py` writes the files under `server/received/`, in the layout of [Data schema §7](02_DATA_SCHEMA.md#7-storage-layout).

The phone talks only to the HTTPS origin it loaded the app from, so there is no CORS or mixed-content problem.

```powershell
# terminal 1, from the repository root: the receiver ("py -3 server/receiver.py --help" lists its options)
py -3 server/receiver.py

# terminal 2: the production build, served with the /api proxy
cd app
npm run build
npm run preview

# terminal 3
cloudflared tunnel --url http://localhost:4173
```

Upload settings for this setup are `"mode": "json_post"` and `"endpoint": "/api/ingest"`. Set them on each phone in the researcher settings (§4.4, item 6) or in the config file.

Make sure you edit the copy that is actually served. `npm run preview` serves `app/dist/config.json`, which is copied at build time, so either edit `app/public/config.json` before `npm run build` or edit `app/dist/config.json` afterwards. `npm run dev -- --host` (tunnel to port 5173) serves `app/public/config.json` directly. If the setting matters for the analysis, give lab sessions their own `study_code`.

Alternative: keep the GitHub Pages app and expose only the receiver.

1. Run `cloudflared tunnel --url http://localhost:8787`.
2. On each lab phone, set the override to `json_post` with `https://<random-words>.trycloudflare.com/ingest`.

*Code:* in `json_post` mode the app sends `Content-Type: application/json`, which triggers a CORS preflight, and the receiver answers it (`app/src/core/uploader.ts`). The hostname changes on every run, so update the phones each time.

After the session, copy `server/received/` into a dated snapshot (§7) and back it up. The folder is git-ignored.

## 6. Recruit participants

### 6.1 Before sharing the link

1. The consent text, `consent_version` and `contact` are final. The participants are the researcher's adult friends, and no ethics review is sought: the researcher's decision ([Considerations §1](06_CONSIDERATIONS.md#1-ethics-and-law)).
2. Uploads reach the service: record one trial on a phone and see it on the admin page ([10](10_FIELD_GUIDE.md#check-uploads-live)).
3. An iPhone and an Android phone each completed a street visit on the live URL ([03 §8](03_SENSORS_AND_PLATFORMS.md#8-pilot-verification-checklist)).

### 6.2 The link

Share the Pages URL as text and as a QR code. Any QR generator works, and Chrome on desktop has one built in. Two optional URL parameters, `?lang=he` and `?lang=en`, set the language and are remembered on that phone. Never share a link with `?sim=1` or `?setup=1`.

The two-sentence explanation for friends, in English and Hebrew, and what to do on the spot are in the [field guide](10_FIELD_GUIDE.md). Tell participants:

- **Open the link in Safari or Chrome**, not inside WhatsApp, Instagram or Facebook. If it opens inside another app, use that app's "Open in browser" option. *Code:* the app detects the in-app browsers it knows (`device.in_app_browser`) and asks the participant to switch. Brave on Android blocks the motion sensors ([Brave][brave-sensors]).
- **Keep the page open** until the Done screen says "All data uploaded ✓".
- **Stay in one browser on one phone.** A second browser or the Home Screen copy starts a new, separate profile ([Data schema §10](02_DATA_SCHEMA.md#10-deduplication-rules)).

### 6.3 Add to Home Screen (optional)

This is recommended for iPhone participants who will come back on another day. Home Screen web apps are exempt from Safari's deletion of site data after 7 days without interaction ([WebKit][webkit-itp-7day]).

| Platform | Steps | Notes |
|---|---|---|
| iPhone (Safari) | Share button > Add to Home Screen > Add | Since iOS 26, a site added to the Home Screen opens as a web app by default ([Safari 26][safari-26-notes]). The Home Screen app has its own storage and permission state, so a profile created in Safari is not visible there. Screen Wake Lock works in Home Screen apps only from iOS 18.4 ([WebKit bug 254545][wk-254545]). On older iOS, use Safari. |
| Android (Chrome) | ⋮ menu > Add to Home screen (or Install app) > Install | Installed sites are more likely to get persistent storage ([web.dev][webdev-persistent]). |

### 6.4 Before each collection day

1. Open the live URL on a test phone and check the app version.
2. Open the service's `/api/health` and check that it answers `"ok": true`, then sign in to the admin page and look at "Refused or ignored uploads" (Apps Script: the `/exec` health route and the Executions page).
3. Charge the phones and turn off Low Power Mode on the iPhones.
4. Plan no deploys for the day.

## 7. Collect the data and run the analysis kit

### 7.1 Get the data

| Source | How | Where to put it |
|---|---|---|
| The Heroku database (this study) | `analysis\.venv\Scripts\python.exe analysis\scripts\pull_live.py --real-only` (credentials from `heroku.env`; [10](10_FIELD_GUIDE.md#pull-the-data)) | `analysis/data/live/`, replaced on every pull; copy it to a dated snapshot before analysing |
| Drive folder (Apps Script) | Download the study folder (the web UI zips large folders into 2 GB parts, [Google][drive-zip]) or sync it with Google Drive for desktop. | A dated, read-only snapshot, for example `analysis/data/real/2026-11-20/` (git-ignored) |
| Index Sheet (Apps Script) | File > Download > CSV | Next to the snapshot. Use it as the inventory and for the weekly recruitment counts. |
| Lab receiver | Copy `server/received/`. | The same snapshot |
| Participant exports | Files named `nosetouch_<participant_id>_<YYYYMMDD-HHMM>.json` (or `.json.txt`), format `nose-touch-export` | An `exports/` sub-folder of the snapshot |

Never edit the snapshot. Deduplicate and filter in code ([Data schema §10](02_DATA_SCHEMA.md#10-deduplication-rules)): keep `simulated == false`, the intended `study_code`, and the latest version of each session.

### 7.2 Set up the analysis kit

```powershell
cd analysis
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. In Git Bash, activate with `source .venv/Scripts/activate`. On macOS and Linux, use `source .venv/bin/activate`.

### 7.3 Run it

Run the pipeline on the synthetic dataset first (§1), so you know every step works before real data arrive. The scripts are in `analysis/scripts/`. Run each with `--help` to see its arguments.

| Script | Purpose |
|---|---|
| `qa_report.py` | Data-quality report: counts per participant and condition, rates and gaps, calibration verdicts, flags |
| `train_baseline.py` | Baseline models (`YawSignRule`, logistic regression, `HistGradientBoosting`) with subject-wise CV |
| `plot_trial.py` | Plots one trial: streams, cues and segmentation. |

The library is `analysis/motionlab`, with the modules `io`, `conventions`, `preprocess`, `segment`, `features`, `augment`, `synth`, `models`, `evaluate` and `qa`. Normalise signs and axes before computing features ([Sensors and platforms](03_SENSORS_AND_PLATFORMS.md)). The methods are described in [Segmentation](04_SEGMENTATION.md) and [Modeling plan](05_MODELING_PLAN.md). Keep the locked test set out of every run until the final evaluation ([Considerations §8](06_CONSIDERATIONS.md#8-statistics)).

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "The app could not start on this browser" | Storage is blocked: a private window or an in-app browser. | Open the link in a normal Safari or Chrome window. |
| "No motion events arrive", or the sensor check finds no data | A desktop browser (no sensors), a page not served over HTTPS, an in-app browser or WebView, Brave or another privacy browser ([Brave][brave-sensors]), or the screen is locked or the app is in the background. | On a desktop, add `?sim=1`. On phones, use the Pages URL or a tunnel, open it in Safari or Chrome, and keep the screen on. |
| Rotation data missing (`rotation_available: false`) | The phone has no gyroscope (some budget Android phones), so the `rotationRate` fields are null ([Chromium][cr-dm-pump-cc]). | Flag the participant. Gyro-based features are not available for that phone. |
| iPhone: motion permission "denied" with no prompt | An earlier Cancel is cached for the site until Safari is quit ([WebKit][wk-permission-cache]; [Apple forum][apple-forum-128376]), or the page is not on HTTPS, which is denied silently. | Follow the steps on screen; they depend on where the app runs. Safari tab: fully close Safari (swipe it away in the app switcher) and reopen the link. Home Screen app: close it in the app switcher and open it again from the Home Screen. Chrome, Firefox or Edge for iOS: close that browser completely. In-app browser: open the link in Safari. Do not use a Private tab mid-session: it has separate, temporary storage, so the participant gets a new code and cannot resume. Deleting the site's website data is not a reliable reset either (WebKit keeps the decision in memory per app process), and it deletes unsent data. |
| Android: motion access blocked | Chrome's site setting for motion sensors. Since Chrome 151 it can be Allow, Ask or Block ([Chrome 151][chrome-151]). | ⋮ > Settings > Site settings > Motion sensors, then allow the site. |
| No sound | The media volume is down, Bluetooth headphones are connected, or audio is not unlocked yet (the first tap unlocks it; [Chrome autoplay policy][chrome-autoplay]). The iPhone ringer switch should not matter, because the app requests the `playback` audio session where supported (Safari 16.4+; [MDN data][bcd-audiosession]). | Raise the media volume, disconnect the headphones and tap again (Continue, or the next block's Start: iOS plays audio only after a tap, and on iPhones that tap also repairs or replaces audio that stayed silent after a call or an app switch). "Play a test beep" on the How-to card checks the sound. The colour cues work regardless. On a computer running the simulator the silence is deliberate. |
| Uploads stay pending or failed, and the console shows `TypeError: Failed to fetch` | Apps Script answered with an HTML error page, which has no CORS header. Causes: the deployment was deleted or not updated, an uncaught script error, a missing re-authorisation, or a quota error. Other causes: a content blocker blocks `script.google.com`, or the phone is offline ([Tanaike][tanaike-webapps]; [content blockers][dwyl-358]). | Open the `/exec` URL (health check) and the Executions page. Redeploy as a new version (§4.3), and run a function in the editor to re-authorise. These failures retry automatically with backoff (*Code:* `uploader.ts`); a refusal by the backend does not (next row). A participant with a blocker can export instead. |
| Uploads are rejected (`bad_study_key` or another `retry: false` answer) and stay failed | The phone still carries an old `study_key` (it was open while you changed the key), or the backend refuses the record. Rejected uploads are not stored, and the app does not retry them on its own. | Reload the app so it reads the current `config.json`, then tap "Upload now" on the Done or Data screen. The reason of each refusal is on the admin page ("Refused or ignored uploads"; Apps Script: the `quarantine` sheet; local receiver: `server/received/quarantine.log`) (§4.4, item 3). |
| Heroku: every upload answers 503 and the phones keep retrying | The database is unreachable or not attached (`DATABASE_URL`), so the service stores nothing on purpose. | `heroku logs --tail -a <app>` (look for `database_missing`), `/api/health`; the phones keep their records and send them once it answers again ([backend/heroku](../backend/heroku/README.md#operations)). |
| CORS error in the console | `json_post` mode is pointed at Apps Script: its `Content-Type: application/json` triggers a preflight that Apps Script cannot answer ([Tanaike][tanaike-webapps]). Or your own `json_post` server does not answer `OPTIONS`. | Use `"apps_script"` mode for Apps Script. On your own server, answer preflights (the receiver does). |
| The `/exec` URL shows a Google sign-in page in a private window | The web app was deployed with "Anyone with Google account" or "Only myself". | Edit the deployment and set Who has access to Anyone (§4.2). |
| "Service invoked too many times", or uploads stop after a busy session | Apps Script quotas: 30 simultaneous executions per user, and daily quotas that reset 24 h after the first request ([quotas][gas-quotas]). Or Drive is full ([Google One][googleone-15gb]). | Let the queue retry, spread lab sessions over time, and free Drive space. |
| Blank page on GitHub Pages | An absolute `base` instead of `'./'`, a failed build, the Pages source not set to GitHub Actions, or a very old browser (the build targets ES2020, meaning iOS Safari 14+ and Chrome 87+; `vite.config.ts`). | Check the Actions run and the browser console for 404s. Restore `base: './'`, set the Pages source, update the browser. |
| An old version keeps loading after a deploy | The service worker serves the cached app shell (production builds register `sw.js`). | Close every tab of the app and reopen it, or reload twice. On a desktop: DevTools > Application > Service workers > Unregister. As a last resort on a phone, clear the site's data (export unsent data first). |
| An edit to `config.json` has no effect | Invalid JSON (phones keep their last loaded configuration, new phones get the defaults with uploads off) or an invalid value (that field falls back to its default), the deploy did not run, the app was not reopened, or the phone was offline and used its saved copy. | Validate the JSON (§4.4), check the Actions run, and reopen the app while online. |
| "Blocked request. This host (…) is not allowed." | The dev or preview server was reached through a host that is not in `allowedHosts`. | Use cloudflared, or add the domain to `tunnelHosts` in `vite.config.ts`. |
| Data recorded through yesterday's tunnel are missing | Each `trycloudflare.com` hostname is a separate origin with its own storage. | Upload or export before you stop a tunnel. |
| iPhone data gone after a week | Safari deletes a site's storage after 7 days of use without interaction. Home Screen apps are exempt ([WebKit][webkit-itp-7day]). | Configure uploads (they run after every trial), suggest the Home Screen app to participants who come back on another day, and use the export. |
| "UnknownError: Connection to Indexed Database server lost" | WebKit's networking process died ([WebKit bug 273827][wk-idb-273827]). | *Code:* the app reopens the database and retries once. If the error persists, reload. |
| Low sample rate (< 50 Hz) in the QA report | iOS Low Power Mode (its effect is undocumented), Firefox on a phone that lacks one of the motion sensors (events at about 10 Hz, inferred from the source; [Firefox][ff-nsdevicesensors]), or a busy page. | Turn off Low Power Mode, use Safari or Chrome, and exclude the trials by the QA rule. |
| The screen rotates during landscape trials | A web page cannot lock the screen orientation on iOS ([MDN data][bcd-screenorientation]). | On iPhone, turn on Rotation Lock in Control Center. On Android, turn off auto-rotate. |
| Trials end as `interrupted` and the session pauses | The page was hidden by a call, a notification, a screen lock or an app switch ([W3C][w3c-do]). A trial that was already recording the return of the arm stays complete (`return_cut_short`, `hidden_after_end`). | Tap Continue. Turn on Do Not Disturb, keep the screen on and keep the app in front. |
| The screen dims or locks during a block; trials carry `wake_lock_not_held` | The screen wake lock was refused (Home Screen apps before iOS 18.4, some in-app browsers) or released. | Set Auto-Lock (iPhone) or Screen timeout (Android) to 5 minutes for the visit, as the How-to card then asks; use Safari or Chrome. |

[wk-permission]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/DeviceOrientationAndMotionAccessController.cpp
[wk-permission-cache]: https://github.com/WebKit/WebKit/blob/main/Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp
[apple-forum-128376]: https://developer.apple.com/forums/thread/128376
[trycloudflare]: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/
[cloudflared-downloads]: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/
[vite-cve]: https://github.com/vitejs/vite/security/advisories/GHSA-vg6x-rcgg-rjx6
[vite-server-options]: https://vite.dev/config/server-options
[chromium-sw-ssl]: https://github.com/endlessm/chromium-browser/blob/aa8c819d5ad2fcb3854a688a0401975eca721f43/content/browser/service_worker/service_worker_consts.cc
[mkcert]: https://github.com/FiloSottile/mkcert
[gh-pages-create]: https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
[gh-plans]: https://docs.github.com/en/get-started/learning-about-github/githubs-plans
[student-pack]: https://education.github.com/pack
[gh-pages-limits]: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
[gh-pages-about]: https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages
[fetch-spec]: https://fetch.spec.whatwg.org/
[gas-content]: https://developers.google.com/apps-script/guides/content
[gas-web]: https://developers.google.com/apps-script/guides/web
[gas-deployments]: https://developers.google.com/apps-script/concepts/deployments
[gas-quotas]: https://developers.google.com/apps-script/guides/services/quotas
[gas-manifest-webapp]: https://developers.google.com/apps-script/manifest/web-app-api-executable
[gas-releases]: https://developers.google.com/apps-script/releases
[tanaike-webapps]: https://github.com/tanaikech/taking-advantage-of-Web-Apps-with-google-apps-script
[labnol]: https://labnol.org/code/20282-publish-google-script-web-app
[drive-limits]: https://support.google.com/drive/answer/37603
[googleone-15gb]: https://support.google.com/googleone/answer/9312312
[drive-zip]: https://workspaceupdates.googleblog.com/2016/08/improvements-to-downloading-files-and-folders-in-Drive.html
[cr-share-allowlist]: https://github.com/chromium/chromium/blob/main/components/browser_ui/webshare/android/java/src/org/chromium/components/browser_ui/webshare/ShareServiceImpl.java
[brave-sensors]: https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620
[webkit-itp-7day]: https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/
[safari-26-notes]: https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes
[wk-254545]: https://bugs.webkit.org/show_bug.cgi?id=254545
[webdev-persistent]: https://web.dev/articles/persistent-storage
[cr-dm-pump-cc]: https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc
[chrome-151]: https://developer.chrome.com/release-notes/151
[chrome-autoplay]: https://developer.chrome.com/blog/autoplay
[bcd-audiosession]: https://github.com/mdn/browser-compat-data/blob/main/api/AudioSession.json
[dwyl-358]: https://github.com/dwyl/learn-to-send-email-via-google-script-html-no-server/issues/358
[wk-idb-273827]: https://bugs.webkit.org/show_bug.cgi?id=273827
[ff-nsdevicesensors]: https://github.com/mozilla-firefox/firefox/blob/main/dom/system/nsDeviceSensors.cpp
[bcd-screenorientation]: https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json
[w3c-do]: https://w3c.github.io/deviceorientation/
