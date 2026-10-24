# Upload backends

The collector app uploads every record (one `SessionRecord` or `TrialRecord`) as an `UploadEnvelope` in its own `POST` (`app/src/core/uploader.ts`, [Data schema §6](../docs/02_DATA_SCHEMA.md#6-uploadenvelope-and-exportbundle)). A backend has to store those records in the shared folder layout ([Data schema §7](../docs/02_DATA_SCHEMA.md#7-storage-layout)) and answer with a short JSON status. Where the app sends is set in `app/public/config.json` (`upload.mode`, `upload.endpoint`, `upload.study_key`) and needs no rebuild.

| Option | Use it for | Status |
|---|---|---|
| [Heroku upload service](heroku/README.md) (`heroku/`) | The v2 study: live uploads into Postgres (`json_post`), admin page, exports for `analysis/scripts/pull_live.py` | Implemented and tested (`backend/heroku/tests`, run with its own venv) |
| [Google Apps Script](google-apps-script/README.md) (`google-apps-script/`) | Data collection without a server of your own | Implemented and tested (`app/tests/appsScript.test.ts`) |
| [Local receiver](../server/README.md) (`server/receiver.py`) | Development, and lab sessions whose data should stay on a laptop | Implemented and tested (`server/test_receiver.py`) |
| Firestore, Supabase, Cloudflare Worker + D1 | Alternatives if Google Drive is not acceptable to the ethics committee | Not implemented; trade-offs below |
| `"mode": "none"` | Before approval, self-tests, or if the committee rejects cloud storage | Built in: participants export a file from the Done screen |

## What every backend must do

These requirements come from the transport and from [Considerations §2](../docs/06_CONSIDERATIONS.md#2-privacy-and-security):

1. **Validate.** At least `kind`, `record.record_type`, the id patterns and the schema major version. The endpoint is public, because it ships in a public web page.
2. **Cap the size** (about 2 MB; a trial is 30–75 KB) and **store nothing of what is rejected.** A rejected upload can hold a participant's profile, and a copy outside `<study_code>/<participant_id>/` would survive a deletion request. Log a metadata line instead (time, reason, size, a SHA-256 of the body and the record's ids), capped per day so that a misconfigured phone or a spammer cannot fill the storage. Both implemented backends cap it at 500 per UTC day.
3. **Overwrite by record id.** The app re-sends a record when an answer was lost, and it re-sends the session record after every trial. Storing the same record twice must give one file, not two.
4. **Answer JSON.** `{"ok": true}` means stored. `{"ok": false, "retry": true}` means try again later (busy, outage). `{"ok": false, "retry": false}` means the record itself was rejected: the app then stops sending that record on its own until the participant taps "Upload now". Never return stored data.

How the app talks to each kind of server:

| `upload.mode` | Request | Suitable servers |
|---|---|---|
| `apps_script` | `POST` with a plain-text body and no custom headers: a CORS "simple request" without preflight. The app counts a record as stored only on a readable `{"ok": true}`. | Apps Script only. It cannot answer CORS preflights. |
| `json_post` | `POST` with `Content-Type: application/json`, which triggers a CORS preflight. A `2xx` answer that is empty or JSON without `"ok": false` counts as stored. | Any HTTPS server that answers preflights: the local receiver, or a Worker you write |

Firestore and Supabase expect their own authentication headers or SDKs, so using them needs a new upload mode in `app/src/core/uploader.ts`.

## Option 1: Google Apps Script web app

The v2 study uploads to the Heroku service instead ([heroku/README.md](heroku/README.md)); Apps Script remains the option that needs no server of your own. A script in a dedicated Google account receives the uploads, writes one JSON file per record into a private Drive folder and logs a row per upload in an index spreadsheet. Setup: [google-apps-script/README.md](google-apps-script/README.md).

- **Pros.** Free, no credit card, no inactivity pausing, about 15 GB of storage (the study needs about 0.25 GB per 100 participants with a `standard` session). The data are plain files that you download as a zip or sync with Drive for desktop. No client SDK, and the whole backend is one commented file of about 730 lines.
- **Cons.** About 30 simultaneous executions per account, so lab bursts must be spread out ([quotas](https://developers.google.com/apps-script/guides/services/quotas)). Errors outside the script's own handling reach the phone as "Failed to fetch". A consumer account offers no choice of data region. The script needs the full Drive scope, which is why it should run in an account used for nothing else.

## Option 2: the local receiver (development and lab sessions)

`server/receiver.py` uses only the Python standard library. It validates like the Apps Script backend and writes the same folder tree under `server/received/`. The phone reaches it through the Vite `/api` proxy (same origin, so no CORS) or through a Cloudflare quick tunnel. Details: [server/README.md](../server/README.md).

- **Pros.** The data never leave the laptop, nothing to set up, and it is ideal for testing uploads end to end.
- **Cons.** It runs only while the laptop does, and a quick-tunnel URL changes on every run. It has no authentication beyond the study key and no rate limiting. Use it for supervised sessions, not as an always-on public server.

## Alternatives (not implemented)

Checked in October 2026. Re-check the limits before you commit to one.

| | Firestore (Firebase Spark plan) | Supabase (Free plan) | Cloudflare Worker + D1 (Free) |
|---|---|---|---|
| Card needed | No | No | No for Workers and D1 (medium confidence); R2 object storage needs a subscription checkout |
| Storage | 1 GiB | 500 MB database, 1 GB file storage | 500 MB per D1 database, 5 GB per account |
| Write limits | 20,000 writes per day, 1 MiB per document | Shared CPU; exceeding 500 MB can restrict the whole organisation (HTTP 402) | 100,000 requests and 100,000 rows written per day, 2 MB per row |
| Pausing | No | Yes: free projects pause after 7 days of low activity | No |
| CORS and auth | SDK or REST with an auth token; anonymous sign-in limited to 100 new accounts per hour per IP | REST with the publishable key in an `apikey` header; answers preflights | You write the CORS and validation code |
| App changes | New upload mode | New upload mode | None: `json_post` mode, if the Worker answers like the receiver |
| Getting the data out | Admin-SDK script (managed export needs the paid plan) | SQL dump or CSV export | `wrangler d1 export` |

**Firestore.** Store one document per trial with the trial id as document id, the metadata as fields and the samples packed into one string field with its index switched off (every array element is indexed by default, and documents are capped at 1 MiB). Allow `create` only, with anonymous sign-in once per device. 1 GiB holds roughly 12,000–30,000 trials. **Cloud Storage for Firebase is not an option without a card:** since 3 February 2026 it requires the paid Blaze plan, even for the default bucket ([Firebase](https://firebase.google.com/docs/storage/faqs-storage-changes-announced-sept-2024); [Firestore quotas](https://firebase.google.com/docs/firestore/quotas); [Auth limits](https://firebase.google.com/docs/auth/limits)).

**Supabase.** One `trials` table with a `jsonb` payload. Two 2026 changes matter:

- New tables are no longer exposed to the Data API automatically. This has been the default for new projects since 30 May 2026, and it applies to all existing projects from 30 October 2026. Every new table needs an explicit `GRANT INSERT ... TO anon`, row-level security switched on and an insert-only policy ([changelog](https://supabase.com/changelog/45329-breaking-change-tables-not-exposed-to-data-and-graphql-api-automatically); [RLS](https://supabase.com/docs/guides/database/postgres/row-level-security)).
- Insert with `Prefer: return=minimal`. Asking for the inserted row back also needs a `SELECT` policy, and an upsert (`ON CONFLICT`) needs `SELECT` rights too, so treat a duplicate-key error as "already stored".

Use the publishable key (`sb_publishable_…`), never the secret or `service_role` key. Free projects pause after 7 days without enough database activity, which would stop uploads during a holiday unless a daily keep-alive job runs ([pausing](https://supabase.com/docs/guides/platform/free-project-pausing)). 500 MB holds roughly 6,000–17,000 trials.

**Cloudflare Worker + D1.** A small Worker can accept exactly the app's `UploadEnvelope`, answer the preflight and return the receiver's JSON, so the app works unchanged in `json_post` mode. Each trial fits in one row. It is the most work of the three: a Cloudflare account, `wrangler`, your own validation and CORS code, and an export step ([Workers limits](https://developers.cloudflare.com/workers/platform/limits/); [D1 limits](https://developers.cloudflare.com/d1/platform/limits/)).

## Recommendation

Use Apps Script for collection and the local receiver for development and supervised lab sessions. If the ethics committee requires that data stay on university infrastructure, ask Bar-Ilan IT for an HTTPS host. A hardened receiver behind a TLS reverse proxy then works in `json_post` mode. Whichever backend you use, keep the export button as every participant's fallback, and deduplicate by record id at analysis time ([Data schema §10](../docs/02_DATA_SCHEMA.md#10-deduplication-rules)).
