# NoseTouch upload service (Heroku + Postgres)

A small web service that receives the collector app's uploads live and keeps them in a Postgres database on Heroku. The app needs no code change: it already has an upload mode, `json_post`, that posts one record per request ([`app/src/core/uploader.ts`](../../app/src/core/uploader.ts)). This service answers that mode exactly like the local receiver ([`server/receiver.py`](../../server/receiver.py)) and the Apps Script backend ([`../google-apps-script/Code.gs`](../google-apps-script/Code.gs)), and it gives the researcher:

- an admin page at `/admin` that refreshes every 15 s (participants, sessions, trials, refused uploads, downloads, deletion);
- `/api/export.zip`, the same folder tree as the other backends, which [`analysis/scripts/pull_live.py`](../../analysis/scripts/pull_live.py) downloads for the analysis kit;
- `/api/export`, the app's own export format;
- [`import_export.py`](import_export.py), which sends export files that participants shared (or that were made before the service existed) through the same door.

Stack: Python 3.12, FastAPI, uvicorn, SQLAlchemy 2 Core, psycopg 3. Without `DATABASE_URL` it runs on SQLite, which the tests use.

## Contents

| Path | What |
|---|---|
| `app.py` | ASGI entry point (`uvicorn app:app`) |
| `nosetouch_server/validation.py` | The upload contract ported from `receiver.py`. Standard library only. |
| `nosetouch_server/store.py` | Idempotent upserts, deletion, admin data |
| `nosetouch_server/exporter.py` | Streamed JSON and zip exports |
| `nosetouch_server/admin.py` | Admin authentication and the `/admin` page |
| `nosetouch_server/db.py`, `settings.py`, `main.py` | Tables, configuration, routes |
| `import_export.py` | Backfill tool for export files |
| `deploy.sh` | Pushes this folder to Heroku without a commit in the main repository |
| `Procfile`, `.python-version`, `requirements.txt` | What Heroku reads |
| `tests/` | pytest suite (`requirements-dev.txt`) |

## Endpoints

| Method and path | Who | What |
|---|---|---|
| `POST /api/ingest` (also `/ingest`) | the app | One `UploadEnvelope`. See [the contract](#the-ingest-contract). |
| `GET /api/ingest` | the app's settings screen | Endpoint probe. It checks the database the way `/api/health` does: 200 `{ok: true, service, server_version, time, configured: true}` when an upload could be stored now; 503 `{ok: false, error, retry: true, service, server_version, time, configured: false}` otherwise, which the app's **Test** shows as an error. |
| `GET /api/health` | anyone | `{ok, service, server_version, time, db: {ok, dialect, schema_version}, counts: {participants, sessions, trials}}`. Answers 503 when the database is unreachable. |
| `GET /` | anyone | One line of text and no data |
| `GET /admin` | admin | The admin page |
| `GET /admin/login`, `POST /admin/login`, `GET /admin/logout` | admin | Browser sign-in and sign-out |
| `GET /api/export` | admin | The app's export format: `{format: "nose-touch-export", schema_version, app_version, exported_at, participants, sessions, trials}`. Gzipped when the client accepts it. |
| `GET /api/export.zip` | admin | `<study_code>/<participant_id>/<session_id>/session.json` and `trial_<index:03d>_a<attempt>_<trial_id>.json` |
| `DELETE /api/participants/{participant_id}` | admin | Deletion on request. Returns `{ok, participant_id, deleted: {participants, sessions, trials, ingest_log}}`. |

Both exports take the filters `?participant_id=P-XXXXXX`, `?since=<ISO time>` and `?include_simulated=false` (the default is `true`). `since` keeps the records whose current version arrived at or after that time, plus the sessions of those trials, so every export can be loaded on its own. Exports are streamed: memory stays flat however big the study gets.

## The ingest contract

The contract below is taken from `uploader.ts` (`sendEnvelope`, `interpretResponse`), `receiver.py` and `Code.gs`.

**Request.** `POST` with `Content-Type: application/json`, and `text/plain` is accepted too, like the receiver. The uploader sends `credentials: 'omit'` and no other header. The body is one envelope `{kind: "session"|"trial", study_key: string|null, sent_at: ISO time, record}`. The study key travels inside the body, never in a header. Bodies over `MAX_BYTES` (default 4,000,000) get 413 `too_large`, also chunked ones without `Content-Length`; nothing beyond the cap is kept in memory. Before answering, the service reads an oversized body to its end and discards it, up to 8 × `MAX_BYTES` (32 MB), like `receiver.py`. A browser sends the whole body before it reads the answer, and a server that closes the connection while body bytes are still arriving makes the operating system reset it. The browser would then report a network error instead of the 413, and the phone would send the record again forever. A body declared larger than 8 × `MAX_BYTES`, or a chunked one that grows past it, is refused unread with `Connection: close`; a browser may only see a network error for it.

**Validation.** These are `receiver.py`'s rules, with its codes and messages:

- `bad_envelope`, `bad_kind`, `bad_study_key`, `bad_record`, `bad_record_type`: the envelope's shape, the study key and the record type.
- `bad_schema_version`: only `1.x` is accepted.
- `bad_study_code`, `bad_participant_id`, `bad_session_id`, `bad_trial_id`, `bad_trial_index`, `bad_attempt`: id patterns. Windows device names and `quarantine` are refused, because export.zip folders are unpacked on Windows.
- `malformed_json`: not valid UTF-8, a BOM, `NaN` or `Infinity`, numbers out of range, or broken JSON.
- `empty_body`, `too_large`.

There is one addition, `unsupported_text`: a NUL character or an unpaired surrogate. Postgres cannot store either, and the app never produces them.

**Answers.** The app reads `ok`, `retry` and `error`.

| Status | Body | The phone then... |
|---|---|---|
| 200 | `{ok: true, kind, id, stored_as, received_at, server_version}`, plus the header `X-NoseTouch-Outcome: stored`, `replaced`, `unchanged` or `kept_newer` | marks the record as sent |
| 400, 403, 404, 405, 413 | `{ok: false, error, code, retry: false, received_at, server_version}` | keeps the record but stops sending it on its own; **Upload now** sends it again. For a body over 8 × `MAX_BYTES` (32 MB) the connection is cut instead (see above), so the phone sees a network error and keeps retrying it. |
| 500, 503 | `{ok: false, error, retry: true, received_at, server_version}` | retries later, with backoff. This includes every upload while the database is unreachable or, on Heroku, not attached (see `DATABASE_URL` below). |

`stored_as` is the record's path inside `export.zip`, the same string `receiver.py` returns.

**Retries and versions.** Every answer to a valid upload is 200, so the phone always stops sending.

- **Sessions.** The app re-sends the session record after every trial. A different version replaces the stored one only if its `updated_at` is later; on equal `updated_at`, the later envelope `sent_at` wins. An older retry that arrives last is ignored (`kept_newer`).
- **Trials.** A `trial_id` is one attempt. An identical re-send changes nothing (`unchanged`). The app legitimately saves an attempt more than once, in this order: a placeholder before the participant answers (`outcome.note = "review_interrupted"` when its review opens in tap mode, `"hand_not_reported"` before the hand question of the v2 choice trial), then the decided copy, and possibly a revoked copy later (`"redo_requested"`, from "Redo last movement" or "Last one was the wrong hand" on the pause sheet). A copy later in that order always replaces an earlier one, never the other way round, whatever the envelope times (log codes `decided_copy`, `revoked_copy`, `review_interrupted_copy`, `kept_revoked`). That is the rule of [docs/02 §10](../../docs/02_DATA_SCHEMA.md#10-deduplication-rules) and of `motionlab.io`. Between copies of the same rank, the later envelope `sent_at` wins. The `review_interrupted` column marks placeholder copies of either kind.
- **Participants.** There is no participant envelope. The profile travels inside every session (`record.participant`, already redacted by the app), and the newest one by its own `updated_at` is kept.

Every upload, accepted or not, adds one row to `ingest_log`. A row holds the time, the outcome, the code, the ids, the size and the SHA-256 of the body. It never holds the body or a name. Rows older than 30 days are pruned, and at most the newest 20,000 are kept. Refusals are logged at most 500 times per UTC day.

**CORS** applies to `/api/ingest`, `/ingest` and `/api/health` only:

- **Preflight from an allowed origin:** status 204 with these headers:
  - `Access-Control-Allow-Origin: <origin>`
  - `Access-Control-Allow-Methods: GET, POST`
  - `Access-Control-Allow-Headers: Content-Type`
  - `Access-Control-Max-Age: 86400`
  - `Vary: Origin, ...`
  - no `Access-Control-Allow-Credentials`
- **Actual requests from an allowed origin:** `Access-Control-Allow-Origin` and `Vary: Origin`.
- **Other origins:** the preflight gets a 403 without CORS headers, and a simple cross-site `POST` gets 403 `bad_origin`.
- **No `Origin` header** (curl, `import_export.py`): accepted.
- **Admin routes:** they never send CORS headers.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | none, so SQLite | Set by the Heroku Postgres add-on (`postgres://...`). It is rewritten to `postgresql+psycopg://`, and `sslmode=require` is added for non-local hosts. **Required on Heroku.** A dyno's memory and disk are wiped at every restart, at least once a day. So when `DYNO` is set and `DATABASE_URL` is not a Postgres URL, the service stores nothing: every upload answers 503 `retry: true`, the phones keep their records, and the startup log has a `database_missing` error line. Otherwise the phones would mark the records as sent after a 200 that the next restart undoes. |
| `ALLOW_EPHEMERAL_DB` | off | `1` lets a dyno use the throwaway SQLite database anyway (a test app only) |
| `SQLITE_PATH` | none, so in memory | SQLite file used when `DATABASE_URL` is unset (local runs) |
| `STUDY_KEY` | none, so any upload is accepted | Must equal `upload.study_key` in `app/public/config.json`. That file is public, so the key is a spam filter, not security. |
| `ADMIN_TOKEN` | none, so admin is off (503) | Long random secret for `/admin`, the exports and deletion |
| `ALLOWED_ORIGINS` | `https://axelfridman.github.io,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173` | Comma-separated origins, or `*` |
| `MAX_BYTES` | `4000000` | Largest accepted request (see [Differences](#differences-from-receiverpy)) |
| `FORCE_HTTPS` | on when `DYNO` is set (on Heroku) | Plain-http `GET` is redirected to https, and other plain-http requests are refused |
| `INGEST_LOG_DAYS`, `INGEST_LOG_MAX_ROWS`, `REJECT_LOG_DAILY_CAP` | `30`, `20000`, `500` | Limits of `ingest_log` |
| `ADMIN_COOKIE_MAX_AGE` | `604800` (7 days) | How long a browser stays signed in |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` | `3`, `2` | Connections per worker process. Essential plans allow 20 (Essential-2 allows 40). |
| `WEB_CONCURRENCY` | uvicorn: 1 | Worker processes. uvicorn reads it. |

## Costs

Prices were checked in October 2026 on [heroku.com/pricing](https://www.heroku.com/pricing) and the [Postgres plans page](https://elements.heroku.com/addons/heroku-postgresql). Heroku bills [to the second](https://devcenter.heroku.com/articles/usage-and-billing), up to the monthly maximum.

| Item | Plan | $/month | Notes |
|---|---|---|---|
| Web dyno | **Basic** | 7 | Always on, 512 MB |
| Web dyno | Eco | 5 | Not recommended. It sleeps after 30 minutes without traffic, and the first upload after that waits for a cold start. |
| Postgres | Essential-0 | 5 | 1 GB, 20 connections |
| Postgres | **Essential-1** | 9 | 10 GB, 20 connections |
| Postgres | Essential-2 | 20 | 32 GB, 40 connections |

**Recommended: Basic and Essential-1, $16 a month.** The cheapest workable setup is Basic and Essential-0, $12 a month.

Measured on the real export of P-VPSDL4: its 23 trials take 1.6 MB as JSON and 1.0 MB in Postgres (JSONB is compressed), about 45 KB per trial.

| Data | Size in Postgres |
|---|---|
| One `quick` session (24 trials) | about 1.1 MB |
| One `standard` session (about 60 attempts) | about 2.8 MB |
| 100 participants with a `standard` session | about 0.3 GB |

Essential-0 is enough for one standard session per participant. Essential-1 leaves room for `full` sessions (about 6 MB each), repeat sessions and simulator data. Essential plans are capped at 99.5% expected uptime, with unannounced maintenance. The phones keep every record and retry, so a short outage loses nothing.

## Deploy

You need the Heroku CLI, logged in (`heroku login`), and Git Bash on Windows. Nothing is committed to the main repository: `deploy.sh` builds a throwaway repository with only this service and force-pushes it.

```bash
APP=nosetouch-study                        # choose a free name
heroku create "$APP"                       # prints the web URL: https://<app>-<hash>.herokuapp.com/
heroku addons:create heroku-postgresql:essential-1 -a "$APP"
heroku addons:wait -a "$APP"               # until the database is provisioned and DATABASE_URL is set
heroku config:set -a "$APP" \
  STUDY_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(18))')" \
  ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
backend/heroku/deploy.sh "$APP"            # --dry-run lists what would be pushed
heroku ps:type web=basic -a "$APP"         # an always-on dyno (not Eco)
heroku apps:info -a "$APP"                 # Web URL
curl -s https://<app>-<hash>.herokuapp.com/api/health
```

Keep the values for the helper scripts in `%USERPROFILE%\.nosetouch\heroku.env` (Windows) or `~/.nosetouch/heroku.env`. This file is never in the repository.

```text
APP_URL=https://<app>-<hash>.herokuapp.com
STUDY_KEY=<the STUDY_KEY above>
ADMIN_TOKEN=<the ADMIN_TOKEN above>
```

`deploy.sh` refuses to push when the app has no `DATABASE_URL` (it asks the Heroku CLI), because the service would then refuse every upload. `ALLOW_EPHEMERAL_DB=1` on the app skips that check. `heroku config -a "$APP"` shows the values again. Redeploying is just running `deploy.sh "$APP"` again. The tables are created at startup if they are missing, and the database row `meta.schema_version` (currently 1) is where later migrations hook in.

### Point the app at it

In `app/public/config.json`:

```json
"upload": { "mode": "json_post", "endpoint": "https://<app>-<hash>.herokuapp.com/api/ingest", "study_key": "<STUDY_KEY>" }
```

Then push the app as usual; GitHub Pages serves the new `config.json` without a rebuild. In the app's researcher settings, **Test** probes the endpoint, and it should answer `ok: true, service: nosetouch-heroku`. Phones that had refused records (for example, ones sent with an old study key) send them again when the participant taps **Upload now**.

## Admin page

Open `https://<app>-<hash>.herokuapp.com/admin/login` and paste the `ADMIN_TOKEN` into the form. The form posts the token, so it never appears in a URL. The browser gets an HttpOnly, Secure, SameSite=Strict cookie that is valid for 7 days. The cookie holds an expiry and an HMAC of it, never the token, and changing `ADMIN_TOKEN` signs every browser out.

You can also sign in with `/admin/login?token=<ADMIN_TOKEN>`. This is handy for a bookmark, but that one URL is written to Heroku's router logs; the service's own access log redacts it. Use the form when in doubt.

The page shows:

- totals;
- one row per participant: code, name or nickname (only if the app uploads one, which `upload_nickname` decides), age, sex, handedness, EHI, phone hand, sessions, trials and accepted trials, left/right counts, platform, first and last upload;
- the latest sessions and trials;
- refused uploads and ignored stale copies;
- download links (all data, real data only, per participant);
- a form to delete a participant.

Every uploaded string is HTML-escaped, and the page loads no external assets (CSP `default-src 'none'`).

## Pull the data for analysis

```powershell
analysis\.venv\Scripts\python.exe analysis\scripts\pull_live.py            # uses heroku.env
analysis\.venv\Scripts\python.exe analysis\scripts\pull_live.py --real-only --participant P-XXXXXX
```

This replaces `analysis/data/live/` (git-ignored) with the current server state, then prints counts per participant. Load it with:

```python
from motionlab.io import load_dataset
ds = load_dataset('data/live')
```

Without the script:

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" -o live.zip https://<app>-<hash>.herokuapp.com/api/export.zip
```

The loader also reads the zip as it is.

## Backfill export files

```powershell
backend\heroku\.venv\Scripts\python.exe backend\heroku\import_export.py nosetouch_P-VPSDL4_20261024-0159.json --url https://<app>-<hash>.herokuapp.com/api/ingest --study-key <STUDY_KEY>
backend\heroku\.venv\Scripts\python.exe backend\heroku\import_export.py export.json --dry-run    # check only
```

The script sends each session and then its trials, in the app's envelope, one request at a time. It retries network errors and 5xx answers with backoff, and it prints a summary table with one line per refused record. It needs only the Python standard library.

`sent_at` is the export's `exported_at`. As a result, a backfill never overwrites a copy that the phone uploaded after the export was made. Running it twice is harmless: the second run answers `unchanged`.

Participants have no envelope of their own. Their profiles arrive inside the session records, and the summary names any profile in the file that no session carries.

## Deletion on request

Use the form at the bottom of the admin page, or:

```bash
curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" https://<app>-<hash>.herokuapp.com/api/participants/P-XXXXXX
```

This deletes the participant, their sessions, their trials (including trials filed under their sessions) and their `ingest_log` rows. Also delete the data on the participant's phone (the Data screen), because records still queued there would be uploaded again. After that:

- delete the participant's folder from every pulled copy (`pull_live.py` replaces `analysis/data/live/`);
- remember that Heroku backups keep a copy until they expire. Essential plans keep daily backups for 7 days and one weekly backup.

## Operations

- **Logs.** `heroku logs --tail -a $APP` shows one JSON line per upload, for example `{"event":"ingest","outcome":"stored","kind":"trial","id":"…","participant_id":"P-…","bytes":84213,"ms":21.4,"request_id":"…"}`. `request_id` is the router's `X-Request-Id`, the same id as in the router's own log line. Lines carry ids and sizes only, never bodies or names.
- **Backups.** Run `heroku pg:backups:capture -a $APP` before risky changes. `heroku pg:backups:schedule DATABASE_URL --at '03:00 Asia/Jerusalem' -a $APP` keeps daily backups. Essential-tier retention: 7 daily, 1 weekly, and 5 manual backups.
- **SQL.** `heroku pg:psql -a $APP`, for example `SELECT participant_id, count(*) FROM trials GROUP BY 1;`. The `record` columns are JSONB, so `record->'condition'->>'hand'` works.
- **Keys.** `heroku config:set ADMIN_TOKEN=<new> -a $APP` restarts the dyno and signs every browser out. A new `STUDY_KEY` must reach the phones through `config.json`. Until a phone reloads the app, its uploads are refused (`bad_study_key`, visible on the admin page) and kept on the phone for **Upload now**.
- **Database outages.** While the database is unreachable, uploads, the app's **Test** probe and `/api/health` answer 503, and the phones retry later. A dyno that starts during an outage waits at most about 25 s for the database. It then opens its port anyway, because Heroku stops a web dyno that has not opened it within 60 s (error R10). It keeps answering 503 and connects as soon as the database is back.
- **Load.** One Basic dyno handles far more than this study needs: a trial upload takes about 20 ms of server time.

## Local development and tests

```powershell
cd backend\heroku
py -3.12 -m venv .venv            # or any Python >= 3.11
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest                       # SQLite, in memory
$env:SQLITE_PATH = "nosetouch.db"; $env:ADMIN_TOKEN = "dev"
.venv\Scripts\python.exe -m uvicorn app:app --port 8000  # http://127.0.0.1:8000/admin/login
```

The same suite runs against Postgres when `NOSETOUCH_TEST_DATABASE_URL` points to a throwaway database whose name contains `test`; its tables are dropped for every test. For example: `NOSETOUCH_TEST_DATABASE_URL=postgres://postgres@127.0.0.1:55432/nosetouch_test`. One test checks that both exports load with `motionlab.io`. It runs inside the analysis venv (`analysis/.venv`), or is skipped with a reason when that venv is missing.

## Differences from receiver.py

| Topic | receiver.py | This service |
|---|---|---|
| Re-sent records | The file is overwritten (last write wins) | The newer version wins (`updated_at` for sessions; for trials the decided copy, then the later `sent_at`). An older copy is acknowledged but ignored, so a slow retry can no longer undo a participant's decision. |
| Size limit | 2 MB | 4 MB. The largest real trial so far is 165 KB (23 s at about 7 KB/s). The wait for stillness has no time limit, and a 413 is a permanent refusal on the phone. Both read an oversized body to its end before the 413, up to 8 × the limit. |
| Chunked bodies | Refused (411) | Read with the cap enforced; an oversized one is read to its end before the 413, up to 8 × the limit |
| CORS | `*` | `ALLOWED_ORIGINS` only, plus 403 `bad_origin` for other sites' browsers |
| Rejection log | `quarantine.log` | The `ingest_log` table, which records accepted uploads too. Its daily cap counts per process, like the receiver's. |
| Extra rejection | none | `unsupported_text` (NUL, unpaired surrogate) |
| Health | `GET /health` | `GET /api/health` with database status and counts. `GET /api/ingest` answers the app's probe like the receiver. |
