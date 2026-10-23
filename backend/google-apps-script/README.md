# Google Apps Script backend: step-by-step setup

This folder is the default upload backend of NoseTouch. It is a free Google Apps Script web app that runs inside the study's own Google account. There is no server to run and no billing account. Every upload from a phone is one record. The script stores each record as one JSON file in a private Drive folder and logs one row per upload in an index spreadsheet.

| File | What it is |
|---|---|
| `Code.gs` | The receiver: `doPost` (uploads), `doGet` (health check), `setup()` and `testIngest()` (run them from the editor) |
| `appsscript.json` | The project manifest: V8 runtime, time zone, web-app access and the two OAuth scopes |

You need about 20 minutes. No programming is required: you copy two files, click Run twice and click Deploy once.

## Before you start

1. **Use a dedicated consumer Google account.** Create a new `@gmail.com` account for the study and turn on 2-step verification. Do not use your private account or a Bar-Ilan Workspace account:
   - the web app runs every upload with the owner's access to all of Drive ("Execute as: Me"), so a separate account limits what a mistake can expose;
   - Workspace administrators can restrict Apps Script and anonymous web-app access, which the app needs (medium confidence, from a [Google Developer Expert](https://labnol.org/code/20282-publish-google-script-web-app)).
2. **Ethics first.** Collect data from anyone but yourself only after ethics approval ([Considerations](../../docs/06_CONSIDERATIONS.md)). Until then, test with your own phone or with `?sim=1`.
3. Use a desktop browser for the steps below. The Apps Script editor does not work well on a phone.

## 1. Create the project and paste the two files

1. Signed in to the study account, open <https://script.google.com> and click **New project**.
2. Click "Untitled project" at the top and rename it, for example to `NoseTouch receiver`.
3. The editor shows a file `Code.gs` with an empty `myFunction`. Select everything in it, delete it, and paste the whole content of [`Code.gs`](Code.gs) from this folder. Save with Ctrl+S (Cmd+S on a Mac).
4. Show the manifest: click **Project Settings** (the gear icon on the left) and tick **Show "appsscript.json" manifest file in editor**.
5. Go back to the **Editor** (the `< >` icon), open `appsscript.json`, replace its content with [`appsscript.json`](appsscript.json) from this folder, and save.

What the manifest sets:

| Setting | Value | Why |
|---|---|---|
| `runtimeVersion` | `V8` | Modern JavaScript; `Code.gs` needs it. |
| `timeZone` | `Asia/Jerusalem` | The study runs in Israel. It only affects how the editor displays times. The receiver writes every time as a UTC ISO string, for example `2026-11-20T09:15:02.123Z`. |
| `webapp.executeAs` | `USER_DEPLOYING` | "Execute as: Me". Participants never sign in or authorise anything. |
| `webapp.access` | `ANYONE_ANONYMOUS` | "Who has access: Anyone". The phone's upload carries no Google login ([manifest reference](https://developers.google.com/apps-script/manifest/web-app-api-executable)). |
| `oauthScopes` | Drive and Sheets | `DriveApp` needs the full Drive scope to create and overwrite files; the index needs Sheets. Nothing else (no e-mail, no external requests). |

## 2. Run `setup()` and grant the permissions

1. In the toolbar, pick **setup** in the function list (next to the Debug button) and click **Run**.
2. The first run asks for authorisation. Click **Review permissions** and choose the study account.
3. Google shows "Google hasn't verified this app". That is expected for a script you wrote yourself. Click **Advanced**, then **Go to NoseTouch receiver (unsafe)**.
4. Google now lists each permission with its own checkbox. **Tick every box** (Drive and Sheets) and click **Continue**. Since 2025 consent is granular ([release notes](https://developers.google.com/apps-script/releases)), and a script missing a scope fails on every upload. If you unticked one by mistake, run `setup()` again and you will be asked again.
5. The **Execution log** at the bottom shows three lines: the URL of the root folder, the URL of the index spreadsheet, and whether a study key is set.

Check the result in [Google Drive](https://drive.google.com): there is a new folder `NoseTouchStudy` with a spreadsheet `NoseTouch index` inside. The spreadsheet has three sheets, `sessions`, `trials` and `quarantine`, each with a header row.

`setup()` is safe to run again at any time. It reuses the folder and the spreadsheet, recreates whatever is missing, and stores their ids in the script's properties:

| Script Property | Set by | Meaning |
|---|---|---|
| `ROOT_FOLDER_ID` | `setup()` | Drive folder that receives all data |
| `INDEX_SPREADSHEET_ID` | `setup()` | The index spreadsheet |
| `MAX_BYTES` | `setup()` (default `2000000`) | Largest accepted request in bytes. A trial is 30–75 KB, a session at most about 30 KB. |
| `STUDY_KEY` | you (optional, §3) | When set, every upload must carry this key |
| `QUARANTINE_COUNT` | the receiver | Rows written to the `quarantine` sheet today (UTC), as `<date>:<count>`, for the cap of 500 per day (§9). Delete it to reset today's count. |

You can see and edit them under **Project Settings > Script Properties**.

## 3. Optional: set a study key

The study key keeps random junk out of your folder. It is a spam filter, not a password: the app sends it from `config.json`, which anyone can download from the public site.

1. Make up a random string, for example by running `node -e "console.log(crypto.randomUUID())"`.
2. In **Project Settings > Script Properties**, click **Add script property**, enter `STUDY_KEY` as the property and the string as the value, and click **Save script properties**.
3. Put the same string into `upload.study_key` in `app/public/config.json` (§6).

Uploads with a missing or wrong key are rejected, and nothing of them is stored. The `quarantine` sheet gets one row per rejection with the reason and the ids of the record (§9), so a phone with an old configuration shows up there instead of failing silently. The phone keeps the record and stops sending it on its own; the participant can still export it.

**Changing the key.** Phones keep the old key until they reload `config.json`, which happens the next time the app is loaded with a connection. Until then every upload they try is rejected without storing anything (the `quarantine` sheet logs at most 500 rejections a day), and their records stay on the phone. After a reload, **Upload now** on the My data screen sends them again; that screen tells the participant so. Change the key between collection days if you can.

## 4. Run the self-test

Pick **testIngest** in the function list and click **Run**. It sends a small session and a small trial through `doPost` and prints the replies. The last log line should be `Self-test passed.` If it reports `Self-test FAILED`, the log line before it says why; the usual cause is that `setup()` did not run or a permission was not granted.

Afterwards delete the test data: the folder `selftest` inside `NoseTouchStudy`, and the rows with study code `selftest` in the `sessions` and `trials` sheets.

## 5. Deploy the web app

1. Click **Deploy > New deployment**.
2. Click the gear icon next to "Select type" and choose **Web app**.
3. Fill in:
   - Description: for example `v1`;
   - Execute as: **Me (your study address)**;
   - Who has access: **Anyone**. Not "Anyone with Google account": that option makes Google ask for a sign-in, which the app's upload cannot do, so every upload would fail.
4. Click **Deploy**. Authorise again if asked, ticking every box.
5. Copy the **Web app URL**. It looks like `https://script.google.com/macros/s/AKfy…/exec`. This is the endpoint for the app. Keep the `/exec` ending; the `/dev` URL works only for the script's editors.

## 6. Point the app at the deployment

Edit `app/public/config.json` in the repository:

```json
"upload": {
  "mode": "apps_script",
  "endpoint": "https://script.google.com/macros/s/<DEPLOYMENT_ID>/exec",
  "study_key": "<the STUDY_KEY value, or an empty string if you set none>"
}
```

Commit and push to `main`; the deploy workflow republishes the site, and phones pick up the change the next time the app starts. Check the JSON before committing: `node -e "JSON.parse(require('fs').readFileSync('app/public/config.json','utf8'))"` prints nothing when the file is valid. More about `config.json` is in [Deployment §4.4](../../docs/07_DEPLOYMENT.md#44-edit-apppublicconfigjson).

`apps_script` mode sends each record as a plain-text `POST` without custom headers. That is a CORS "simple request": the browser sends no preflight, which Apps Script could not answer. Do not use `json_post` mode with Apps Script.

## 7. Smoke tests

In PowerShell type `curl.exe`, because plain `curl` is an alias of `Invoke-WebRequest` there. Replace `<URL>` with your `/exec` URL.

1. **Health check.** Open the URL in a browser, or run:

   ```powershell
   curl.exe -sL "<URL>"
   ```

   Expected: `{"ok":true,"service":"nosetouch-receiver","server_version":"1.0.0","time":"…","configured":true}`. A Google sign-in page means "Who has access" is not "Anyone". `"configured":false` means `setup()` has not run.
2. **CORS headers.**

   ```powershell
   curl.exe -s -D - -o NUL -H "Origin: https://example.github.io" "<URL>"
   ```

   Expected: a `302` with `access-control-allow-origin: *` and a `location:` header that points to `script.googleusercontent.com`. On macOS or Linux use `-o /dev/null`.
3. **An upload that must be rejected.** It proves that `doPost` runs and answers JSON. In PowerShell, write the body to a file first, because quoting JSON for `curl.exe` is error-prone:

   ```powershell
   Set-Content -Encoding ascii ping.json '{"ping":true}'
   curl.exe -sL -H "Content-Type: text/plain" --data-binary "@ping.json" "<URL>"
   ```

   On macOS or Linux: `curl -sL -H 'Content-Type: text/plain' --data '{"ping":true}' "<URL>"`. Expected: `{"ok":false,"error":"kind must be 'session' or 'trial'","code":"bad_kind","retry":false,…}`. The `quarantine` sheet now has a `bad_kind` row, and no file was written; you can delete the row. (`-L` follows the redirect as a GET, as a browser does, which is how Apps Script delivers the answer.)
4. **End to end.** Record one trial on a phone. The Done screen should say that everything was sent, and the file should appear in Drive under `NoseTouchStudy/<study_code>/<participant>/<session>/`.

Errors of the script itself show up in the editor under **Executions** (the list icon on the left).

## 8. Update the code without changing the URL

1. Paste the new `Code.gs` into the editor and save.
2. Click **Deploy > Manage deployments**, select the active deployment and click **Edit** (the pencil icon).
3. Set **Version** to **New version** and click **Deploy**.

The `/exec` URL stays the same, so `config.json` needs no change ([deployments](https://developers.google.com/apps-script/concepts/deployments)). Saving alone does not change what the deployment runs, and **New deployment** would create a new URL. The health check shows `server_version`, so you can confirm which version is live.

- If a new version of the script uses an additional Google service, run any function once in the editor and tick the new permission. Until you do, every upload fails, and the app only sees "Failed to fetch".
- A script keeps at most 200 versions. If you ever get close, delete old versions under **Project history**.
- Older versions of the script stored rejected payloads as files in `NoseTouchStudy/quarantine/`. If that folder exists, delete it: its files can hold whole records, profiles included. Then delete the `quarantine` sheet and run `setup()` once, which recreates the sheet with the current columns.

## 9. What the receiver does

Every request gets a short JSON answer. The app reads it to decide whether a record is safely stored:

| Answer | When | What the phone does |
|---|---|---|
| `{"ok": true, "kind", "id", "stored_as", "received_at", "server_version"}` | The record was written to Drive | Marks the record as sent |
| `{"ok": false, "error", "code", "retry": false}` | Rejected: not JSON, wrong ids or schema version, wrong study key, larger than `MAX_BYTES` | Marks the record as refused but keeps it, and goes on with the other records. It does not send a refused record again on its own: **Upload now** on the My data screen does, for example after a reload picked up a fixed `config.json`. The receiver stores nothing of the payload; the `quarantine` sheet gets one row. |
| `{"ok": false, "error": "busy", "retry": true}` | Too many uploads at the same moment (a lock wait timed out) | Retries later |
| `{"ok": false, "error": "…", "retry": true}` | `setup()` has not run, or Drive or Sheets failed | Retries later |

Storage layout, shared with the local receiver and the synthetic data ([Data schema §7](../../docs/02_DATA_SCHEMA.md#7-storage-layout)):

```text
NoseTouchStudy/
  NoseTouch index                         spreadsheet: sessions, trials, quarantine
  <study_code>/<participant_id>/<session_id>/
    session.json                          the SessionRecord, replaced on every re-send
    trial_<trial_index, 3 digits>_a<attempt>_<trial_id>.json
```

- **Validation.** `kind` is `session` or `trial` and matches `record.record_type`; `schema_version` starts with `1.`; `study_code`, `session_id` and `trial_id` match `[A-Za-z0-9_-]{1,64}`; `participant_id` matches `P-` plus 6 characters without 0, O, 1 and I; `trial_index` and `attempt` are whole numbers. The study code `quarantine` is reserved.
- **Rejections leave no file.** A rejected upload can still contain a participant's profile, and a copy outside `<study_code>/<participant_id>/` would survive a deletion request. So nothing of it is stored in Drive. The `quarantine` sheet gets one row with `received_at`, `reason`, `bytes`, `sha256` (of the body, to recognise repeats of the same payload) and the identifiers found in the envelope: `kind`, `record_type`, `study_code`, `participant_id`, `session_id`, `record_id`, `schema_version` and `app_version`. It never holds the record or any profile value.
- **At most 500 quarantine rows per UTC day.** The count is kept in the Script Property `QUARANTINE_COUNT`. Beyond the cap a rejection is still answered with `{"ok": false, "retry": false}`, but nothing is written that day, so a misconfigured phone or a spammer cannot fill the spreadsheet.
- **Re-sends are safe.** The app re-sends a record when an answer was lost, and it re-sends the session record after every trial. The receiver then overwrites the same file.
- **The index is a log.** Every upload appends a row, so a session appears once per re-send and a re-sent trial appears twice. Deduplicate by `session_id` or `trial_id` and keep the last row ([Data schema §10](../../docs/02_DATA_SCHEMA.md#10-deduplication-rules)). Text cells are stored as plain text, so nothing a phone sends can become a formula. Raw sensor samples never go into the sheet.
- **Locks.** Folder creation and index rows are written under the script lock, as Google recommends for concurrent writes ([LockService](https://developers.google.com/apps-script/reference/lock/lock)); the slow file write is not.

## 10. Quotas and limits

| Limit | Value | What it means here |
|---|---|---|
| Simultaneous executions | about 30 per account | With "Execute as: Me" all participants count against the owner. An upload takes 1–3 s, and each phone sends one at a time. |
| Run time | 6 minutes per execution | Far above what one upload needs |
| Lock wait | 10 s (folders), 30 s (index), 5 s (quarantine row) | Beyond that the answer is `busy` and the phone retries. A rejection stays a rejection; only its quarantine row is skipped. |
| Request size | `MAX_BYTES`, default 2 MB | Google documents no limit of its own |
| Drive storage | up to 15 GB, shared with Gmail and Photos | A `standard` session is about 2–2.5 MB, so 100 participants need about 0.25 GB. Rejected uploads take no Drive space. |
| Spreadsheet | 50,000 characters per cell, 20 million cells | The index stores ids and metadata only |
| Daily quotas | reset 24 h after the first request | "Service invoked too many times" means wait |

Sources: [Apps Script quotas](https://developers.google.com/apps-script/guides/services/quotas), [Drive file limits](https://support.google.com/drive/answer/37603), measurements of a web app's concurrency by [Tanaike](https://github.com/tanaikech/taking-advantage-of-Web-Apps-with-google-apps-script). Before the first lab session, test with about 30 parallel uploads ([Deployment §4.2](../../docs/07_DEPLOYMENT.md#42-deploy-the-web-app)).

## 11. Download the data

| What | How |
|---|---|
| Everything | In Drive, right-click `NoseTouchStudy` (or one study-code folder) and choose **Download**. Drive zips it; large folders arrive as several zip files of up to 2 GB. |
| A synced copy | Install [Google Drive for desktop](https://www.google.com/drive/download/), sync the folder, and copy it from there. |
| The inventory | Open `NoseTouch index`, then **File > Download > Comma-separated values**. This exports the open sheet; repeat it for each sheet. |

Copy downloads into a dated, read-only snapshot such as `analysis/data/real/2026-11-20/` (git ignores `analysis/data/`). The analysis kit reads the folder tree directly, zip files included. See [Deployment §7](../../docs/07_DEPLOYMENT.md#7-collect-the-data-and-run-the-analysis-kit).

## 12. Privacy and security

- **Keep the folder private.** Never share `NoseTouchStudy` with "Anyone with the link". Add the PI by their named Google account.
- **The endpoint is public by design.** The app is a public web page, so anyone can find the `/exec` URL and the study key. The receiver therefore validates every envelope, caps the size, stores nothing of what it rejects (one metadata row in the `quarantine` sheet, at most 500 a day), never runs anything it receives, and answers with a short status only, never with stored data.
- **Editors see everything.** Because the web app executes as the owner, anyone who can edit the script can read all the data. Do not add editors who may not read the data.
- **Spam.** If junk rows appear in the `quarantine` sheet, set a new `STUDY_KEY` in the Script Properties and in `config.json` (§3), then delete those rows. Rejected uploads never reach Drive. Phones that still have the old key are rejected the same way until they reload the app, and they keep their records for **Upload now**.
- **Deletion requests.** When a participant asks for deletion, delete the folder `<study_code>/<participant_id>/` and the participant's rows in the `sessions`, `trials` and `quarantine` sheets (search for the participant code; the quarantine rows hold only identifiers). Delete the participant from any downloaded snapshot too ([Considerations §1](../../docs/06_CONSIDERATIONS.md#1-ethics-and-law)).
- **Data location.** A consumer Google account offers no choice of data region. Ask the ethics committee whether that is acceptable ([Considerations](../../docs/06_CONSIDERATIONS.md)). The local receiver (`server/receiver.py`) is the alternative that keeps lab data on a laptop.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The app shows "Failed to fetch" and uploads stay pending | Open the `/exec` URL. An HTML error page instead of JSON means the deployment was deleted, not updated after an edit, or is missing a permission. Check **Executions**, redeploy a new version (§8) and run a function in the editor to re-authorise. |
| The `/exec` URL shows a Google sign-in page | "Who has access" is not "Anyone". Edit the deployment (§8) and change it. |
| `"configured": false`, or uploads answer "receiver not configured" | Run `setup()` (§2). The phones keep their records and send them afterwards. |
| Every upload is rejected with `bad_study_key` | `STUDY_KEY` and `upload.study_key` in `config.json` differ. Make them identical, or delete the property to switch the check off. Then reload the app on the phones and tap **Upload now** on the My data screen: the app does not re-send refused records on its own. |
| Many `busy` answers | Too many phones upload at the same moment. Records are retried automatically; spread lab sessions out. |
| `Self-test FAILED` | Read the log line above it. Usually `setup()` did not run or a permission is missing. |
