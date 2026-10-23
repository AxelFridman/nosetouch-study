# Local receiver

`receiver.py` stores the collector app's uploads on your own computer. Use it during development and in supervised lab sessions whose data should stay on the lab laptop, with no Google account involved. It speaks the same protocol as the [Apps Script backend](../backend/google-apps-script/README.md): one `UploadEnvelope` per `POST`, the same validation and the same JSON answers. It writes the same folder tree, so the analysis kit reads both alike ([Data schema §7](../docs/02_DATA_SCHEMA.md#7-storage-layout)):

```text
server/received/
  <study_code>/<participant_id>/<session_id>/
    session.json                                  the SessionRecord, replaced on every re-send
    trial_<trial_index, 3 digits>_a<attempt>_<trial_id>.json
  quarantine.log                                  one line per rejected upload: metadata only
```

A rejected upload is never stored. `quarantine.log` gets one JSON line per rejection with the same fields as the `quarantine` sheet of the Apps Script backend: `received_at`, `reason`, `bytes`, `sha256` (of the body; `null` when the body was too large to read) and the identifiers found in the envelope (`kind`, `record_type`, `study_code`, `participant_id`, `session_id`, `record_id`, `schema_version`, `app_version`). It never holds the record or any profile value. At most 500 lines are written per UTC day; the count starts again when the receiver restarts. The analysis kit does not read `.log` files.

It needs Python 3.10 or newer and nothing else (standard library only). Git ignores `server/received/`.

## Start it

From the repository root:

```powershell
py -3 server/receiver.py            # Windows
python3 server/receiver.py          # macOS / Linux
```

It prints where it listens and where the files go, then one line per request. Stop it with Ctrl+C.

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8787` | TCP port. The Vite `/api` proxy expects 8787. |
| `--host` | `127.0.0.1` | Interface. The Vite proxy and cloudflared connect locally, so the receiver need not listen on the network. |
| `--out` | `server/received` | Output folder |
| `--key` | none | Study key that every upload must carry, like `STUDY_KEY` in Apps Script. Set it whenever a tunnel makes the receiver reachable from the internet. |
| `--max-bytes` | `2000000` | Largest accepted request. A trial is 30–75 KB. |
| `--quiet` | off | No line per request |

`py -3 server/receiver.py --help` lists the same options.

## With the dev server on a laptop (simulated participant)

The app's dev server forwards `/api/*` to `http://127.0.0.1:8787/*` (`app/vite.config.ts`), so the app talks to its own origin and needs no CORS at all.

```powershell
# terminal 1, repository root
py -3 server/receiver.py

# terminal 2
cd app
npm run dev
```

Point the uploads at the proxy. Either set them in `app/public/config.json`:

```json
"upload": { "mode": "json_post", "endpoint": "/api/ingest", "study_key": "" }
```

or, without touching the file, open the app's researcher settings and set the per-device upload override to mode `json_post` and endpoint `/api/ingest`. Then open <http://localhost:5173/?sim=1> and run a `quick` session. Each trial appears under `server/received/` a moment after it ends, and the receiver's console shows `trial <id> -> <path>`.

Do not push a `config.json` with `json_post` and `/api/ingest` to `main`: every push to `main` redeploys the participants' app, and `/api/ingest` exists only on your dev server.

## With phones, through a Cloudflare quick tunnel

Phones need HTTPS for the motion sensors. A quick tunnel gives the laptop a public HTTPS address with a valid certificate and needs no account ([Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)). Install it once with `winget install --id Cloudflare.cloudflared` (macOS: `brew install cloudflared`).

**A. Tunnel the app; uploads go through its proxy (recommended).** The phone loads the app from the tunnel and posts to `https://<tunnel>/api/ingest` on the same origin, so there is no CORS and no mixed content.

```powershell
# terminal 1, repository root
py -3 server/receiver.py --key lab-2026-11

# terminal 2: the dev server ...
cd app
npm run dev -- --host
# ... or, for several phones, offline use and Add to Home Screen, the production build:
#   npm run build
#   npm run preview

# terminal 3: port 5173 for the dev server, 4173 for the preview
cloudflared tunnel --url http://localhost:5173
```

Open the printed `https://<random-words>.trycloudflare.com` address on each phone. Set the upload override (or `config.json`) to mode `json_post`, endpoint `/api/ingest`, and the same study key. `npm run preview` serves the copy of `config.json` made at build time (`app/dist/config.json`), so edit that copy or rebuild after changing `app/public/config.json`.

**B. Keep the deployed app and tunnel only the receiver.** Use this when the phones should run the GitHub Pages version.

```powershell
py -3 server/receiver.py --key lab-2026-11
cloudflared tunnel --url http://localhost:8787
```

On each lab phone, set the override to mode `json_post`, endpoint `https://<random-words>.trycloudflare.com/ingest`, and the study key. In `json_post` mode the app sends `Content-Type: application/json`, which makes the browser send a CORS preflight first; the receiver answers it (`Access-Control-Allow-Origin: *`).

Both ways, the tunnel hostname changes on every run, and a new hostname is a new origin with empty storage on the phone. Upload or export what the phones hold before you stop a tunnel, and clear the overrides after the session ([Deployment §5](../docs/07_DEPLOYMENT.md#5-local-receiver-for-lab-sessions)).

## Check that it works

```powershell
curl.exe -s http://127.0.0.1:8787/
```

The answer is `{"ok": true, "service": "nosetouch-receiver", "server_version": "1.0.0", "time": "…", "configured": true}`. Through the dev-server proxy, the same check is `http://localhost:5173/api/`.

A test upload that must be rejected (PowerShell; on macOS or Linux use `curl -s -H 'Content-Type: application/json' --data '{"ping":true}' http://127.0.0.1:8787/ingest`):

```powershell
Set-Content -Encoding ascii ping.json '{"ping":true}'
curl.exe -s -H "Content-Type: application/json" --data-binary "@ping.json" http://127.0.0.1:8787/ingest
```

It answers `{"ok": false, "error": "kind must be 'session' or 'trial'", "code": "bad_kind", "retry": false, …}` and appends a `bad_kind` line to `server/received/quarantine.log`.

## Answers

| HTTP status | Body | Meaning for the app |
|---|---|---|
| 200 | `{"ok": true, "kind", "id", "stored_as", "received_at", "server_version"}` | Stored; the record is marked as sent. |
| 400 | `{"ok": false, "error", "code", "retry": false, "received_at", "server_version"}` | Rejected: not JSON, wrong ids or schema version, wrong study key. The phone keeps the record but stops sending it on its own. After the cause is fixed (for example the study key on the phone), reload the app and tap **Upload now** on the My data screen. |
| 413 | same, `code: "too_large"` | Larger than `--max-bytes` |
| 404, 405, 411, 501 | same shape | Wrong path, method, or a request without `Content-Length`. The app treats them like a rejection too. |
| 500 | `{"ok": false, "error": "server error: …", "retry": true}` | The receiver could not write (disk full, permissions). The phone retries. |
| 204 | empty | Answer to a CORS preflight (`OPTIONS`) |

Accepted routes: `POST /`, `/ingest` and `/api/ingest` (in case a proxy keeps the prefix). A `GET` on any of these paths, or on `/health`, is the health check, just as Apps Script answers a `GET` on its `/exec` URL; the endpoint test in the app's researcher settings relies on that. The validation rules are those of the Apps Script backend: `kind` is `session` or `trial` and matches `record.record_type`; `schema_version` starts with `1.`; the ids match the schema patterns; `trial_index` and `attempt` are whole numbers. A re-sent record replaces its file atomically (a temporary file, then a rename), so a reader never sees half a file and retries are safe.

## Differences from the Apps Script backend

- **No index spreadsheet.** The folder tree is the data, and the console prints one line per request. Rejections go to `quarantine.log` instead of the `quarantine` sheet; its daily cap is counted in memory, so it starts again when the receiver restarts.
- **Real HTTP status codes.** Apps Script always answers 200, with the outcome in the JSON.
- **Stricter input.** The receiver also rejects bodies that are not valid UTF-8, `NaN` or `Infinity` tokens, and numbers too large for a double. It refuses Windows device names (`CON`, `NUL`, `COM1`, …) as `study_code` or `session_id`, because Windows cannot create such folders. The app never produces any of these.

## Security

The receiver listens on `127.0.0.1` only. A tunnel makes it reachable from the whole internet for as long as it runs, and anyone who learns the URL can post to it. So for a tunnelled session, start it with `--key`, use the same key on the phones, and stop the tunnel when the session ends. The receiver has no other authentication and no rate limiting. It is a tool for supervised sessions, not a server to leave running. After the session, copy `server/received/` into a dated, read-only snapshot such as `analysis/data/real/2026-11-20/` (git-ignored) and back it up.

To delete a participant's data, delete `<study_code>/<participant_id>/` in `server/received/` and in every snapshot, and the participant's lines in `quarantine.log` (search for the participant code; those lines hold identifiers only).

## Tests

```powershell
analysis\.venv\Scripts\python.exe -m pytest server -q
```

`test_receiver.py` starts a real receiver on a free port in a background thread. It covers the layout, idempotent and concurrent writes, every rejection code, the quarantine log (no body, identifiers only, the daily cap), the size limit, CORS and the command-line options.
