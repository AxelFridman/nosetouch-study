# Stage 3: the production page (predict mode)

**Link: https://axelfridman.github.io/nosetouch-study/?predict=1**

The link works once this code is on `main`: `.github/workflows/deploy.yml` then builds and publishes the app. The
same app, opened without `?predict=1`, is still the data collector, unchanged.

The page answers one question: which hand brought the phone to the nose? It shows **P(right)**, the probability that
it was the right hand, computed on the phone from the motion sensors alone. Nothing about the person is asked, nothing
is stored, and nothing leaves the phone.

## How to use it (three lines)

1. Open the link in **Safari** (iPhone) or **Chrome** (Android) and tap **Start** (iPhone: allow motion access).
2. Hold the phone out, screen towards you, then bring it to your nose with one hand and back; repeat if asked.
3. Read **P(right) = N %** (the probability that the RIGHT hand was used). **Again** starts over; the chip at the top
   picks the model; **Compare models** shows all four.

What the page can show:

| Screen | Meaning |
|---|---|
| **P(right) = 97 %**, "probability that the RIGHT hand was used", a LEFT/RIGHT bar, "RIGHT hand · 1 touch" | the answer: the model reached 90 % one way or the other |
| the same with "Leaning RIGHT · not confident · 5 touches" | a forced answer: 5 touches without reaching 90 % (lane m4's rule always answers then) |
| the same, muted, "after 1 touch · not final" | provisional: keep going |
| "Not counted – try again" (small, under the dots) | the last movement was a return, a fidget or a doubtful approach |
| "No touch found – bring the phone to your nose, then back" | no touch was found within 3 minutes |
| "Turn the screen towards your face" | the phone was held with the screen away from the face (or down) before the first touch |

The page finds the touches on its own (no start cue) and decides when to stop; once an answer is shown it stops the
sensors and lets the screen sleep.

URL options (combine with `&`):

| Option | Effect |
|---|---|
| `&model=cnn` (default), `m4`, `phys6`, `yaw` | the model whose answer is shown and whose stopping rule ends the session |
| `&view=compare` | open with the compare view |
| `&lang=he` | Hebrew, right to left |
| `&sim=1` (`&hand=left/right`, `&seed=n`) | desktop demo: a simulated person, replayed live through the same engine. Defaults to `&model=m4`; the CNN is marked "n/a (sim)" there (see the known limits) |
| `&debug=1` | attempts with every model's P, ignored movements, sample rate, the longest engine call and the largest sample gap, thresholds |

The footer shows the model, the first 8 characters of its file's sha256, and the app version with the build's commit
(`cnn · eeb74a79 · v0.3.0+abc1234`): it tells which deploy a phone has loaded.

## The models

All four were fitted on all 20 persons (593 trials) with the Stage 2 code. Each has its own calibrator and its own way
of combining touches. The accuracies are Stage 2 leave-one-person-out results.

| id | Model | Stage 2 name | Accuracy (95 % CI) | Combining k touches |
|---|---|---|---|---|
| `cnn` | Final CNN (default): 4-member 1D CNN, Platt | `m3_cal_deep_platt` | 0.893 (0.850–0.931), one stored window | sigmoid(Σ logits / τ_k) |
| `m4` | Detector model: lane m4's yaw logistic, nested isotonic | `m4_cal_nested` | 0.728 (0.689–0.766), **one recording end to end, cue-free** | sigmoid(s_k · mean logit) |
| `phys6` | Physics logistic (yaw, roll, lateral) | `m1_lr_phys6_nested` | 0.833 (0.792–0.876), one stored window | sigmoid(Σ z / k^0.25) |
| `yaw` | Yaw reference: logistic on yaw at 8 instants | `example_yaw_logreg` | 0.779 (0.731–0.829), one stored window | mean of P (not calibrated for k ≥ 2) |

The movement detector, the attempt rule, the grip check and the stopping thresholds are **lane m4's**:
- answer once max(P, 1 − P) ≥ 0.9;
- at most 5 touches;
- after 5 touches, answer whatever the P (shown as "Leaning …").

These were fitted for lane m4's own classifier and are applied unchanged to every model. Lane prod, which would refit
them for the final CNN, has not run. One change for the page: the grip gate is at rest_up_z −0.05 instead of lane m4's 0
(`analysis/stage3/runtime.py` `PRODUCTION_GRIP_GATE_UPZ`). Correctly held recordings refused / screen-away versions of
real recordings refused: at 0, 27 of 620 / 63 of 64 (26 of 396 on Android: a near-vertical hold sits at about 0); at
−0.05, 8 of 620 / 62 of 64; at −0.1, 4 of 620 / 58 of 64. A screen-away hold that passes is mostly answered with the
wrong hand, while a refusal only asks for a retry, so the page takes −0.05.

## How it works

```
devicemotion -> LiveBuffer (the recorder's rounding) -> 60 Hz stream (stream_from_record)
  -> activity |w| -> hysteresis bouts -> each bout frozen 1 s after it is confirmed
  -> window (17 x 64) -> segment classifier: approach / return / other
  -> every approach = one attempt -> all four models -> each model's P_k -> the driver's stopping rule
```

Cost on a phone: the engine runs on the sensor events, one tick per 100 ms of data.
- A tick reuses the previous tick's activity signal and grid step (exact, bit for bit) and computes the rest pose only
  when a bout is about to be frozen, so it no longer sorts and refilters the whole stream every time.
- The four models of a new attempt (the CNN is most of the cost) run once the phone is still again, at most 1.5 s
  after the freeze, not during the next movement. The window is cut at the freeze either way, so every number is the
  same; only the moment the answer appears moves (by at most 1.5 s).

Code:
- `app/src/predict/`: `pipeline/`, `detect/`, `models/` (JSON bundled into the chunk, no runtime fetch), `live/`, `ui/`.
- `app/src/main.tsx` loads it with a dynamic `import()` only on `?predict=1`, after a `connect-src 'none'` CSP that
  blocks every connection. Predict mode skips `config.json`, the collector's IndexedDB, the uploader and the service
  worker registration.

Specs: `docs/stage3/APP_SPEC.md` (the page) and `docs/stage3/PORT_SPEC.md` (the Python to port).

## Parity with Python

The TypeScript engine reproduces `analysis/stage3/reference.py`, which wraps Stage 2's `predict_stream` and the models.

`app/src/predict/parity.test.ts` replays every fixture through the page's own engine:
- sample by sample, through `createLiveSession` and the wired binding;
- with each of the four models as the driver;
- comparing the stream, bouts, windows, every model's logits and P(right), the k-aggregations, the compare view and the
  decisions.

| Check | Cases | Result |
|---|---|---|
| `parity.test.ts`, synthetic fixtures (committed, CI) | 5 × 4 drivers | windows 0 float32 ulp, \|ΔP\| ≤ 2.2e-16, decisions identical |
| `parity.test.ts`, real fixtures (local only) | 40 × 4 drivers | the same |
| all local recordings (scratch sweep, see below) | 1005 cases × 4 drivers | no difference |
| `live/liveCost.test.ts` | 17 fixtures | the cached tick = the batch activity bit for bit; deferred scoring = scoring at once |
| `e2e/predict.spec.ts` | 2 fixtures in the browser | the page shows Node's P(right) bit for bit (driver and every compare row) |

Last full sweep (0.3.0, after the cached tick; re-run with the −0.05 gate): all 598 study recordings, 22 natural-use ones,
223 excluded attempts, 64 screen-away versions, 97 five-touch sessions and one 3-minute stream, each replayed through
the page's engine with each of the four models deciding (4020 runs). No run differs from the Python reference: windows 0
float32 ulp, worst |ΔP(right)| 2e-15, every bout, attempt, grip refusal and decision identical; the 1118 final
decisions equal Python's live and offline decisions. The sweep scripts are not in the repository (they read every
recording); the committed parity tests above run in CI.

## Running and testing (from `app/`)

| Command | What it does |
|---|---|
| `npm run dev`, then http://localhost:5173/?predict=1&sim=1 | desktop demo |
| `npm run dev -- --host` + `cloudflared tunnel --url http://localhost:5173` | phone over HTTPS (motion sensors need a secure context) |
| `npm run typecheck` | |
| `npm test` | collector tests plus predict mode's (`src/**/*.test.ts`). The real-recording blocks run only where `analysis/data/stage3_fixtures/` exists, and skip in CI |
| `npx playwright test e2e/predict.spec.ts` | the page at phone sizes (Pixel 7, iPhone): demo, injected sensors, compare, Again, permission, layout, Hebrew |
| `npm run build` | `dist/assets/mount-*.js` is the predict chunk: about 1.09 MB raw, 0.47 MB gzip. The collector's `index-*.js` contains none of it |
| `npx tsx scripts/predictReplay.ts src/predict/__fixtures__/syn01_android_right_3touch.json.gz [model]` | one fixture through the engine in Node |

Release checklist (CI runs only `npm test`): `npm run typecheck && npm test && npm run build && npx playwright test`
in `app/`, and `.venv/Scripts/python.exe -m pytest stage3/tests -q -p no:cacheprovider` in `analysis/`.

## Regenerating the models and the fixtures (from `analysis/`)

| Command | Writes |
|---|---|
| `.venv/Scripts/python.exe -m stage3 export` | `app/src/predict/models/*.json` and `manifest.json` (sha256 of each file). Fits are cached in `analysis/data/stage3_cache/`; `--refit` refits (about 30 s) |
| `.venv/Scripts/python.exe -m stage3 fixtures` | the synthetic fixtures (committed) in `app/src/predict/__fixtures__/`; the real-recording fixtures in `analysis/data/stage3_fixtures/` (git-ignored, **never commit**) |
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest stage3/tests -q` | reference = `predict_stream`, JSON = fitted models |

Then run `npm test` in `app/`: the hash test and the parity tests must pass on the new files.

## Known limits

- **Not validated end to end for the CNN.** The detector, the stopping thresholds and the grip check were fitted for
  lane m4's classifier, and the CNN's τ_k on stored windows. Lane prod, which would check the CNN on detected windows,
  has not run. On the 40 local real fixtures (whole recordings) the CNN answered 30 and named the right hand in 27.
- **The demo is not evidence.** The simulator's touches fool the CNN (it names the wrong hand with high confidence),
  so `&sim=1` defaults to m4 and marks the CNN "n/a (sim)". Test the CNN on a real phone.
- **The detector misses some touches.** On the 598 single-touch study recordings, lane m4's detector found no attempt
  in 15 % and 2–3 attempts in 11 % (identical in Python). Expect an occasional "Not counted – try again".
- **The grip check sees only the start.** It looks at the rest before the first movement. A touch made later with the
  screen turned away only shows a hint. It can still refuse a correct, nearly vertical hold (4 of 620 recordings):
  tilt the top of the phone slightly away and tap **Start over**.
- **About 1–2.5 s delay per touch.** A touch counts 1 s after it is confirmed (so the zero-phase filter cannot change
  it), and its models run once the phone is still, at most 1.5 s later.
- **Speed on slow phones.** The engine runs on the main thread, inside the sensor events. On this desktop a tick costs
  about 0.3 ms early and 1.7 ms (median) at 3 minutes, and scoring an attempt (four models, the CNN most of it) 20–50
  ms; a mid-range phone is perhaps 3–6 times slower. A slow phone can therefore delay or drop a few samples around a
  scoring. `&debug=1` shows the longest engine call and the largest sample gap: a gap above 100 ms becomes a hole in
  the stream. Not yet measured on real phones.
- **First load needs the network** (about 0.5 MB gzip; the page shows "Loading…" meanwhile). Offline use works only
  after one online predict visit on a phone that already has the collector's service worker; predict mode never
  registers it.
- **Compare vs selecting a model.** The compare view shows every model after the same touches as the shown answer;
  selecting a model instead applies that model's own stopping rule, which may stop at an earlier touch (so the number
  can differ).
- **Not covered by a fixture.** PHYS6's "unavailable" path (screen exactly horizontal) is covered by unit tests only.
- **Manual checks on real phones.** The iOS permission prompt, the wake lock and the Android vibration must be checked
  by hand.
- **Version.** `APP_VERSION` is 0.3.0; the collector writes it into its records made after this deploy.
