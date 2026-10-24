# 06 · Considerations and risk checklist

This checklist covers what can invalidate the data, the analysis or the PI's evaluation even when the code works. It applies to the street protocol (protocol 2.0.0, app 0.2.0, schema 1.2.0, consent `2026-10-v3`). Read it before the first friends record, then again before the bulk of the collection.

**Related:** [Project overview](00_PROJECT_OVERVIEW.md) · [Protocol](01_PROTOCOL.md) · [Data schema](02_DATA_SCHEMA.md) · [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md) · [Segmentation](04_SEGMENTATION.md) · [Modeling plan](05_MODELING_PLAN.md) · [Deployment](07_DEPLOYMENT.md) · [Stage 3 predictor](08_STAGE3_PREDICTOR.md) · [Stage 2 and 3 plan](09_STAGE2_3_PLAN.md) · [Field guide](10_FIELD_GUIDE.md) · data contract [`schema/records.schema.json`](../schema/records.schema.json)

**Tags**

| Tag | Meaning |
|---|---|
| **P0** | Silently invalidates data or exposes participants. Do it before recording anyone. |
| **P1** | Do it before the bulk of the collection, after the first friends. |
| **P2** | Good practice. Do it when time allows. |
| *Decision* | A choice the researcher has made. Recorded here, not re-opened. |
| *Fact* | A platform, legal or literature statement, with a source. |
| *Code* | Behaviour of this repository. The file is named. |
| (no tag) | Our recommendation. |

## 0. Critical path (every P0 item)

1. Record only adults who tick the in-app consent. No ethics review is sought: the participants are the researcher's adult college friends (§1).
2. Keep the consent text, `consent_version` and `min_age: 18` in step with what participants are told (§1).
3. Keep the names and the admin token private: names only in the private database and the raw records, the token only in `heroku.env` (§2).
4. Treat the upload endpoint as public: validation, a size cap, no storage of refused uploads, an origin allowlist (§2).
5. Show and repeat the safety wording: firm grip, gentle touch, stop on pain (§3).
6. Keep wrong-hand and wrong-grip trials visible. Never drop them silently and never keep them silently (§4).
7. Record both hands for every participant, and ask left-handed friends on purpose (§5, §9).
8. Take the iOS sign inversion and the `rotationRate` axis mapping from the data. Never assume them (§6).
9. Never commit raw data, never edit it, and back up the database (§7).
10. Evaluate by person: person-wise CV, a locked test set and person-level CIs (§8).
11. Rehearse the demo on the kind of phone the PI is likely to use (§11).

## 1. Ethics and law

1. **P0 · No ethics review: the researcher's decision.** *Decision* (2026-10-24): this collection does not go to an ethics committee. The participants are the researcher's own college friends: adults who volunteer, read the consent on the app's first screen and tick it before anything is recorded. What follows from the decision:
   - only adults (`min_age: 18`), only after the in-app consent, only motion data;
   - the practical safeguards of §2 apply to every session: names on a private server, the admin token kept secret, deletion on request;
   - the decision covers this admission project; whether the data may be reused, for example in a publication, is a question for the PI (§12).

2. **P0 · The consent text.** *Code:* the welcome screen shows four key facts, a "Full information" section and the checkbox, and every `SessionRecord` stores `consent.version` and `consent.accepted_at` (`app/src/i18n/en/onboarding.ts`, `app/src/app/sessionFlow.ts`). What version `2026-10-v3` tells participants:

   | Element | What the app states |
   |---|---|
   | Purpose | Researchers (or "Researchers at {institution}") study whether a phone's motion sensors can tell if it is held in the left or the right hand; the recordings develop and test such a method. |
   | Procedure | About 30 movements, about 40 with the optional bonus; both hands; upright and sideways, screen facing the participant; different speeds and a few different movements; any part of the phone may touch the nose; no table or other equipment. |
   | Duration | About 5 minutes, plus an optional bonus of about 1 minute. Pause or stop at any time; everything recorded is kept. |
   | Risks | Dropping the phone (hold it firmly), bumping the nose (touch gently, also in the "fast" movements); stop at once if anything hurts or feels uncomfortable. |
   | Voluntary, withdrawal | Stop at any moment without a reason. Delete the data on the phone on "My data". Uploaded data are deleted on request, quoting the participant code shown at the end. |
   | Data recorded | Motion sensors (accelerometer, gyroscope, orientation and, where present, the compass direction) and screen touches, only while a movement is recorded; name, age, sex, dominant hand, usual phone hand, height (optional); phone and browser details. |
   | Never recorded | Camera, microphone, location, contacts, anything from other apps. |
   | Names | Stored with the data on the private research server so the researcher can find the recordings, for example to delete them; never published. |
   | Storage | The phone and a private research server run by the research team; each movement is sent as soon as it is recorded. |
   | Retention | As long as the research and its publications need it, under the institution's rules; publications never contain names. |
   | Eligibility | The checkbox: "I agree, I'm 18 or older, and I have no arm or hand injury that makes this uncomfortable." `region` in `config.json` adds "I'm taking part from …". |
   | Contact | `contact.name` and `contact.email` from `config.json`; while they are empty: "contact the researcher who sent you this link". |

   Still open: fill `contact` (and `institution`, if wanted) in `config.json`; have a native speaker proofread the Hebrew text; decide whether to mention that GitHub Pages and Heroku (its router log and the service's access log) see IP addresses ([GitHub Pages][gh-pages-about]). No log holds names or record bodies.

3. **P0 · Version the consent text.** Bump `consent_version` in `config.json` (and `CONSENT_VERSION_DEFAULT` in `app/src/core/constants.ts`) whenever the wording changes. *Code:* the value is copied into each session, so you can tell which text each participant accepted.

4. **P0 · Adults only.** *Code:* `min_age: 18` in `config.json` gates the profile, and the consent checkbox repeats it. Age is self-reported, so the gate filters but does not verify.

5. **P1 · Amendment 13 to the Privacy Protection Law.** *Fact:* in force since 14 Aug 2025. It broadens "personal information" to include online identifiers and adds "information of special sensitivity", which includes biometric identifiers used for computerised identification ([IAPP][iapp-a13]; [Pearl Cohen][pearl-a13]; [Library of Congress][loc-a13]). The law defines "biometric" as "a unique human, physiological or behavioral characteristic that can be computationally measured" ([translation][a13-translation]), and arm movements are individual enough to serve as one ([Conti et al. 2011][conti-2011]). With names stored next to the motion data, the database is clearly personal information. So:
   - minimise: age instead of birth date, no location, no contact details in the app;
   - never use the motion data to identify people, and say so if the PI asks for a data plan;
   - keep the database private (§2).

   Registration (s.8a(a)(1)) applies only to databases whose main purpose is passing personal information to others as a business, with data on more than 10,000 people, or to public bodies; a database of special-sensitivity information on more than 100,000 people must notify the Authority. A study of a few dozen friends is far below both.

6. **P1 · GDPR.** *Fact:* the GDPR applies to non-EU controllers that monitor the behaviour of people in the EU (Art. 3(2)(b); [GDPR][gdpr]), including monitoring through smart devices ([EDPB 3/2018 summary][edpb-3-2018]). Record friends located in Israel; `region` can add the location to the consent checkbox.

7. **P1 · Data location.** The data live in a Heroku Postgres database. Heroku apps run in the region chosen at `heroku create` (the United States unless `--region eu` was given); check with `heroku apps:info -a <app>`. Ask the PI whether that is acceptable ([Q2](#12-open-questions-for-the-pi)). Lab sessions can keep their data on a laptop with `server/receiver.py` ([Deployment §5](07_DEPLOYMENT.md#5-local-receiver-for-lab-sessions)).

8. **P1 · Retention and deletion.** Agree a retention period with the PI. When a participant asks for deletion (by code, or by name, which the admin page lists):
   1. delete them on the admin page ("Delete a participant") or with `DELETE /api/participants/<code>` and the admin token. *Code:* one transaction removes the participant, their sessions, their trials and their `ingest_log` rows ([backend/heroku README](../backend/heroku/README.md#deletion-on-request));
   2. ask them to delete the data on their phone too (welcome screen › My data › "Delete this participant and their data"): records still queued there would otherwise upload again;
   3. pull again (`pull_live.py` replaces `analysis/data/live/`) and delete the participant's folder from every snapshot and derived copy, and from lab receiver folders;
   4. remember the database backups: Heroku's Essential plans keep 7 daily backups and 1 weekly, so a copy survives until they expire;
   5. log the date and the code, and nothing else.

   The field steps are in [10](10_FIELD_GUIDE.md#delete-a-participant-on-request). Deleting inside the app (`LocalStore.deleteParticipant`, `clearAll`) removes only the copies on that phone; the consent says so. For the Apps Script backend, delete the folder `<study_code>/<participant_id>/` and the participant's rows of the index sheet ([backend README](../backend/google-apps-script/README.md)).

9. **P2 · Future use.** Decide early whether the data may be reused: publication, later lab projects, open data. Consent is hard to extend afterwards.

## 2. Privacy and security

1. **P0 · Names are stored by design, and kept private.** *Code:* the profile's full name is stored in `nickname`, and `redactProfile` uploads it only when `upload_nickname` is true, which this study's `config.json` sets and the consent states (`app/src/core/config.ts`). Names therefore sit in the database, on the admin page and in raw downloaded records. Rules:
   - `participant_id` stays the key everywhere; names never enter analysis tables (*Code:* `motionlab` drops them), reports or the repository;
   - share derived tables only without names and without `device.user_agent` (`analysis/data/README.md`);
   - report tables use P-codes only and suppress cells of fewer than 3 people ([09 D31](09_STAGE2_3_PLAN.md#6-report-and-deck)).

2. **P0 · The admin token opens everything.** *Code:* `ADMIN_TOKEN` protects `/admin`, both exports and deletion; without it the admin routes answer 503. The login form posts the token, and the browser keeps only an HttpOnly, Secure, SameSite=Strict cookie for 7 days that holds an HMAC, never the token (`backend/heroku/nosetouch_server/admin.py`). Rules:
   - keep it only in `%USERPROFILE%\.nosetouch\heroku.env` (outside the repository) and in Heroku's config vars;
   - never paste it into a chat, an e-mail, a screenshot or a URL you share; prefer the login form to `/admin/login?token=…`, which lands in Heroku's router logs;
   - sign in only on your own devices;
   - if it may have leaked: `heroku config:set ADMIN_TOKEN=<new> -a <app>` (this signs every browser out), then update `heroku.env`.

3. **P0 · The upload endpoint is public.** *Fact:* `config.json`, which holds the endpoint and `study_key`, is a public file, and a GitHub Pages site is public ([GitHub][gh-pages-about]). *Code:* the upload service (`backend/heroku/`):
   - validates every envelope with the rules of `server/receiver.py`: kind, record type, id patterns, schema major version, study key; it also refuses NUL characters and unpaired surrogates;
   - caps bodies at 4 MB (`MAX_BYTES`);
   - never stores a refused body: `ingest_log` keeps the time, outcome, code, ids, size and a SHA-256, at most 500 refusals per UTC day, and prunes rows after 30 days;
   - answers browsers only from `ALLOWED_ORIGINS` (by default the Pages origin and localhost);
   - answers with a short status object, never with stored data.

   The study key is a spam filter, not security. If spam appears, rotate it: a new `STUDY_KEY` on Heroku and the same value in `upload.study_key`. A phone that still has the old page open gets `bad_study_key`; the app keeps those records, does not retry a refusal on its own, and sends them after a reload with "Upload now".

4. **P0 · A private database.** Only the Heroku account that owns the app can reach the Postgres add-on (`DATABASE_URL`). Keep multi-factor authentication on for that account, and add the PI as a collaborator only if they need raw access.

5. **P1 · No location, camera or microphone, and drop the heading.** *Code:* the app asks only for motion and orientation access (`app/src/core/sensors.ts`). Several streams carry a heading: `orientation.compass` (iOS), `orientation_abs` and `quaternion_abs` (Chrome on Android) and the magnetometer where a browser exposes it. A heading is not a location, but it partly identifies the place, carries no left/right information and cannot be mirrored. Exclude these streams from features and from any dataset shared outside the lab ([Q10](#12-open-questions-for-the-pi)).

6. **P1 · Device details are personal data too.** The user agent, `model_hint`, screen size, language and memory partly identify a person, and Amendment 13 counts online identifiers as personal information. Keep them for device analyses; coarsen them to platform, browser and major version before sharing outside the lab.

7. **P2 · Transport and repository hygiene.** *Code:* `config.ts` accepts only `https://` endpoints or same-origin paths, and the service redirects or refuses plain HTTP on Heroku (`FORCE_HTTPS`). `.gitignore` excludes `data/`, `analysis/data/*`, `analysis/reports/*`, `server/received/` and `.env*`; `heroku.env` lives in the user profile, outside the repository.

8. **P1 · Treat downloaded records as untrusted text.** Anyone who finds the endpoint and the public study key can post a schema-valid but fabricated record, and free-text fields such as `quality.warnings`, `outcome.note` or `nickname` can then carry markup or spreadsheet formulas. *Code:* the admin page HTML-escapes every uploaded string and loads no external assets; `qa_report.py` and `train_baseline.py` formula-escape the CSVs they write (`motionlab.report.csv_safe`). Still filter every analysis to the intended `study_code` and `simulated == false`, and import other CSV exports as text.

## 3. Participant safety and comfort

1. **P0 · Firm grip, gentle touch.** *Code:* the consent's risks say "You might drop the phone: hold it firmly." and ask for a gentle touch "also in the "fast" movements"; the How-to card says "Touch gently and hold the phone firmly."; the intro of every block with fast trials says "FAST still means a gentle touch." (`app/src/i18n/en/`). For researchers, the setup screen adds the general safety line and the tap-review Ready screen the fast instruction ("… Touch your nose gently!"). *Fact (pilot):* the fast trials hit the nose at 6.5 and 8.4 g (single-sample spikes when the nose met the screen). Say "fast but soft" aloud if a friend hits hard.
2. **P0 · Stop on pain.** The consent checkbox confirms no arm or hand injury that makes the movement uncomfortable (`arm_issue` is written as false). Anyone may stop at any moment: "Hold to pause", then "Finish now" and "Tap again to finish". Nothing recorded is lost.
3. **P1 · Glasses.** A fast reach can hit the frame. Suggest taking them off or approaching gently.
4. **P1 · Fatigue.** *Code:* GO comes after 400 ms of stillness and a random 0.4–0.9 s foreperiod (`DEFAULT_SEGMENTATION_PARAMS`), later only while a new instruction is read (up to 2.5 s after the trial starts, or as long as the spoken words), the visit takes about 5 minutes, the bonus is optional, and every block intro is a rest. An interrupted session can be resumed: the welcome screen offers it for 24 h (`findResumableSession`), and the profile's Continue resumes it by itself within 2 h (`RESUME_ON_CONTINUE_MS`); after 24 h the next app start closes it as `abandoned`.
5. **P1 · Space and posture.** Clear space around the arm; never while walking, near traffic or on stairs. Sitting is fine: the How-to card's "I’m sitting" switch records it.
6. **P2 · Odd variants.** `lean_in`, `elbow_high` and `jerky` are unfamiliar movements. *Code:* the street visit runs them upright at natural speed, never asks for `chin` or `forehead`, shows each instruction on the live panel, and the pause sheet's "Redo last movement" repeats one.
7. **P2 · Hearing and language.** *Code:* every cue is both a sound and a large colour change, and the screen always faces the participant (no screen-away grips), so the visual cues are always visible. The UI is in English and Hebrew (`?lang=he`, right-to-left).

## 4. Label quality

The label is the hand of the plan (`hand_source = instructed`), or the reported hand of the free trial. *Fact:* in a supervised 7-day wrist-sensor study, 7 of 45 participants (15.6%) wore a sensor on the wrong wrist on at least one day ([PLOE, Sensors 2019][ploe-2019]). Expect 5–15% protocol deviations. In the hands-free flow, `accepted` means "completed, grip not wrong, not taken back", not "confirmed by the participant".

| Problem | Prevention and detection in the app (*Code*) | Offline check |
|---|---|---|
| Wrong hand | One hand per block; the intro says "Phone in your LEFT hand" (and "Switch hands!"); its Start button names the hand; the live panel shows LEFT or RIGHT in huge letters; the pause sheet's "Last one was the wrong hand" re-saves the trial with `participant_flag = wrong_hand` and runs it again. | Wrong-hand screening by an antisymmetric model inside the training folds ([09 D6](09_STAGE2_3_PLAN.md#42-labels-and-exclusions)): trials with p(label) < 0.05 and runs of 3 or more reversed trials. Flags remove trials from training only, never from evaluation. |
| Wrong grip | The grip gate holds GO back while gravity confidently shows another grip and names the instructed grip in words (the grip picture only when the page has turned with the phone); after 25 s without GO the pause sheet names the grip; a completed trial with `orientation_match = false` is auto-rejected (`auto_reject:orientation`) and re-run with the grip hint. | Gravity cannot tell screen toward from away in upright and upside-down poses (`app/src/core/orientationDetect.ts`); screen-away grips are no longer asked for, and touches from the nose suggest the screen faced the participant. |
| Reported hand (free trial) | "Which hand did you use?"; "Not sure" keeps the trial as not accepted with `other_problem` (`hand_unsure`). A copy saved before the answer, and a failed attempt, carry `hand_not_reported` and the plan's placeholder hand. | Choice trials never enter training or the primary metric; only their accepted copies are scored, separately ([09 D4](09_STAGE2_3_PLAN.md#42-labels-and-exclusions)). |
| Moving before GO | GO only after 400 ms of stillness plus a random 0.4–0.9 s; a counted false start (error beep, `timing.false_starts`) only when the motion exceeds 30 deg/s or 0.6 m/s² within 300 ms of losing stillness; sway restarts the hold silently. | Exclude trials with a short or noisy baseline. |
| No real movement, or no hold at the nose | `no_movement` or `too_long`: kept, not accepted, and run again automatically; a second failure in a row opens the pause sheet. | Re-segment offline. Flag disagreements over 100 ms and movements with several peaks ([Segmentation](04_SEGMENTATION.md)). |
| Speed not as instructed | Instruction only (the speed word on the panel; spoken before every trial, except on iPhones, which speak only at a block's Start tap). | Use the measured duration as the analysis variable. |
| Start-position drift | Instruction only. | Track the baseline gravity direction and the movement amplitude per person across `trial_index`. |
| Odd variant not performed | Instruction only. | Count movement units (peaks of ‖ω‖). |
| Interrupted recording | A hidden page, sensors silent for 2 s or no GO within 25 s pause the session; an attempt that reached GO is kept as `aborted` or `interrupted` (one stopped during the return of the arm stays `complete`, `return_cut_short`); `quality.hidden_during` (a hide during the movement; a later one adds `hidden_after_end`), `quality.gaps_over_50ms`, `quality.motion_hz`; `wake_lock_not_held` and `screen_rotated_during_movement` in `quality.warnings`. | Exclude trials hidden during the movement (rule T4), with a gap over 50 ms (T6) or a rate under 50 Hz (T5). In the pilot every such gap began within 15 ms of a screen touch. |
| Simulated data | `simulated: true` on every record made with `?sim=1` (never remembered across reloads) or with `npm run synth`. | Assert `simulated == false`; `pull_live.py --real-only`. |

Watch for protocol drift: plot each person's duration, peak ‖ω‖ and false starts against `trial_index`. Write the exclusion rules down before you see any model result ([02 §12](02_DATA_SCHEMA.md#12-data-quality-and-exclusion-rules)).

## 5. Confounds

| Confound | Why it matters | Handled by the design (*Code*) | Still to do |
|---|---|---|---|
| Hand dominance | *Fact:* about 10.6% of people are left-handed ([Papadatou-Pastou 2020][papadatou-2020]), so for most people "right hand" also means "dominant hand". Dominant and non-dominant reaches differ in curvature and initial direction ([Sainburg 2002][sainburg-2002]), and less so in left-handers ([Przybyla 2012][przybyla-2012]). | Both hands for everyone; the self-reported dominant hand and phone hand in the profile; mirror augmentation and antisymmetric models ([Modeling plan](05_MODELING_PLAN.md)). | At least 8, ideally 10 or more left-handers ([09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection)). Model `hand × handedness`. Report left-handers separately. |
| Start pose | The start sets how large the lateral component is; likely a main driver of errors. | `start_position`: street main trials start `natural`; `shoulder` and `center` are drawn as variety cells. | Report accuracy per start position. Agree the canonical pose with the PI ([Q5](#12-open-questions-for-the-pi)). |
| Instructed vs achieved speed | People read "fast" differently. A 5%-of-peak threshold cuts 5.9% of a minimum-jerk duration off each end: 18 ms at 0.3 s, 119 ms at 2 s ([minimum-jerk model][minjerk-smooth]). | Per-speed segmentation overrides (`resolveParams`); raw streams kept. | Analyse by measured duration. |
| Order, fatigue, learning | Behaviour changes with time in the session. | The street tiers run in a fixed order (upright speeds, sideways, varied movements, bonus) with both hands balanced at every tier boundary; the first hand and the first sideways side follow seed bits; trials are shuffled within blocks. | Tier and condition are confounded with time in the session: use `trial_index` as a covariate, compare hands within a tier, and check the realised split of `protocol.first_hand`. |
| Bonus self-selection | Only some friends opt into tier 4. | `PlannedBlock.optional`; status `completed` does not need it. | Treat bonus trials as extra data, not as a balanced factor. |
| Device model, mass, case | *Fact:* device differences degrade motion classifiers ([Stisen 2015][stisen-2015]); phone mass shifts the tremor spectrum ([phone vs wearable][a20-tremor]). | `device.model_hint` (the model on Android, only the device class on iOS). Protocol 2.0.0 no longer asks for the phone model or a case; the analysis proxies iOS models by screen size (`motionlab.io.device_model`). | Leave-one-device-model-out. |
| Platform quantisation and sign | *Fact:* Chrome rounds values to 0.1 m/s² and 0.1 °/s, WebKit does not ([Chromium][cr-round]); iOS reports acceleration with an inverted sign ([WebKit][wk-motion-mm]). | `calibration.gravity_sign` from the first still upright hold, confirmed offline by the engine and the trial votes; raw values kept; both hands on the same phone. | Quantise iOS data in preprocessing (ablation A3). Leave-one-platform-out. At least 25–30% iPhone users. |
| OS and browser updates during the study | *Fact:* Chrome 151 added `requestPermission()` and plans to make sensor access ask by default ([Chrome 151][chrome-151]); iOS 26 freezes the OS version in the user agent ([Safari 26][safari-26-notes]); a WebKit fix of the sign inversion could ship without notice. | Engine detection and the trial-vote check in every session; `app_version`, `device.browser`, `device.os_version` stored. | Run QA by date and watch the sign source, rate and quantisation. Treat `os_version` as unreliable on iOS 26 and later. |
| Posture | Set per session, so confounded with the person. | `posture` stored; the "I’m sitting" switch. | Use it as a covariate. |
| Setting (street, remote, lab) | Supervision changes behaviour. | n/a | Note the setting in the session log; give a distinct wave its own `study_code`. |
| Browser "acceleration" source | *Fact:* Chrome falls back to a crude gravity estimate when the phone has no linear-acceleration sensor ([Chromium fusion][cr-linacc-fusion]). | `segmentation.acc_source` stored. | Compute gravity and linear acceleration offline from `accg_*` and `rr_*`. |

## 6. Sensor and platform pitfalls

This is a summary. Mechanisms, measurements and the full sources are in [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md).

| Pitfall | Consequence | How it is handled |
|---|---|---|
| About 60 Hz maximum in every browser: WebKit uses a 1/60 s timer ([WebKit][wk-motion-h]), Chromium a 60 Hz pump ([Chromium][cr-pump]), the Generic Sensor API caps at 60 Hz ([Chromium][cr-traits]). | A web page cannot get a higher rate. | 60 Hz is adequate: 99% of the acceleration energy of movements lasting 0.25 s or more lies below about 10 Hz ([model][minjerk-smooth]). A phyphox cohort could validate it ([phyphox forum][phyphox-rates]). |
| iOS reports `acc_*` and `accg_*` with all axes inverted: face-up z ≈ −9.81 ([W3C list 2014][w3c-2014-sign]; [2026 measurement][gaitscope-96]). | On chiral features, iOS trials look like the other hand. | The engine's expected sign, confirmed by the upright-hold votes of the session's trials (`norm_v2`, source `engine_vote`); sessions without a supported sign are excluded (rule S3). Never infer iOS from `requestPermission`, which Chrome 151 has too. |
| `rotationRate` alpha/beta/gamma are rates about device x/y/z ([W3C CRD 2025][w3c-do]; [PR #43][w3c-pr43]); MDN's alpha page still says z. | Gyro channels mislabelled, the mirror wrong. | No tilt step any more: the analysis takes the mapping whose gravity closure over the session's trials is decisive, else assumes W3C and flags the session ([03 §5](03_SENSORS_AND_PLATFORMS.md#5-normalisation-layer-for-analysis)). |
| `event.interval` is ≈ 0.0167 (seconds) on iOS, 16 on Chrome and 100 on Firefox ([WebKit][wk-dme-cpp]; [Chromium][cr-dm-pump-cc]; [Firefox][ff-nsdevicesensors]). | Wrong time steps. | Only per-sample times are used; `interval_raw` is metadata. |
| Euler angles hit gimbal lock at the upright start pose ([W3C][w3c-do]). | alpha and gamma jump. | The orientation stream is metadata only. |
| Chrome quantisation: 0.1 m/s², 0.1 °/s ([Chromium][cr-round]). | A platform shortcut for the model. | Quantise iOS data and stratify by platform. |
| iOS permission needs a tap, a Cancel stays cached until Safari is fully closed, insecure pages are denied silently ([WebKit][wk-permission]; [WebKit cache][wk-permission-cache]). | Sessions with no data. | The permission is requested inside the Start tap; the repair screen gives steps that match the browser; the field guide has them too ([10](10_FIELD_GUIDE.md)). |
| No events arrive while the page is hidden ([W3C][w3c-do]). | Gaps. | The running trial stops and the session pauses; a screen wake lock (the How-to card asks for a 5-minute Auto-Lock when it is refused, and such trials carry `wake_lock_not_held`); the How-to card asks for Do Not Disturb. |
| Safari deletes a site's storage after 7 days without interaction; Home Screen apps are exempt ([WebKit][webkit-itp-7day]). | Unsent data are lost. | Upload right after every save, a keepalive flush when the page is left, and the export button. |
| In-app browsers and WebViews behave unpredictably; Brave on Android blocks the sensors ([Brave][brave-sensors]). | Odd permission behaviour or no data. | `device.in_app_browser` detection and a request to open the link in Safari or Chrome; the repair screen names privacy browsers. |
| The page rotates in sideways grips; iOS cannot lock the orientation ([MDN data][bcd-screenorientation]). | Layout changes mid-trial; possible sample gaps. | The device frame is unaffected; rotations are recorded (`streams.screen_events`), and one between GO and the end adds `screen_rotated_during_movement`; the layout fits landscape; the grip picture shows only when the page turned with the phone. Check `gaps_over_50ms` around screen events on real phones. |
| Screen touches can stall the motion stream (pilot: every gap over 50 ms began within 15 ms of a touch). | Rule T6 dropped 2 of 18 accepted pilot trials. | Watch the T6 rate in QA; sideways grips cause finger touches. |
| Fast trials can go past ±4 g on some phones and reach 500–700 °/s ([Galaxy A20 study][a20-tremor]); the pilot showed 6.5–8.4 g spikes at the nose. | Clipping. | Flag it in QA (rule T10). |
| Vibration contaminates the accelerometer. | An artefact at GO. | *Code:* the phone vibrates only for `block_start` at a block intro (`app/src/core/cues.ts`). |
| A computer running the simulator or tests could beep or talk, or a test could upload to the live database. | Noise next to the researcher; fake records among real ones. | *Code:* the cue sink is silenced in simulation mode and in automated browsers; Playwright runs headless with `--mute-audio`. The local journeys (`npm run e2e`) serve `config.json` with uploads off and block, and fail on, every request to another host (`app/e2e/fixtures.ts`); only the deployed-site smoke test `npm run e2e:live` uploads, and its records carry `simulated: true`. |

Higher rates need native code. iOS apps get at most 100 Hz ([Apple DTS][apple-dts-100hz]); Android apps get up to 200 Hz, or more with a special permission ([Android][android-sensors]). Since May 2026 Expo Go cannot run other people's projects ([Expo][expo-go-may-2026]), so a native arm needs store or TestFlight distribution. Decide on it only after a validation cohort ([Q8](#12-open-questions-for-the-pi)).

## 7. Data management

1. **P0 · Never commit data.** *Code:* `.gitignore` excludes `data/`, `analysis/data/*` (except its README), `analysis/reports/*` and `server/received/`. Live data come from the database with `pull_live.py` into `analysis/data/live/`; dated snapshots go under `analysis/data/` ([Deployment §7](07_DEPLOYMENT.md#7-collect-the-data-and-run-the-analysis-kit)).
2. **P0 · Raw data are read-only.** Never edit downloaded JSON. Put corrections (relabels, exclusions, person aliases) in versioned tables keyed by `trial_id` or participant code, with a reason and a date.
3. **P0 · Back up.** The Heroku database is a single point of failure. Schedule daily backups (`heroku pg:backups:schedule DATABASE_URL --at '03:00 Asia/Jerusalem' -a <app>`; Essential plans keep 7 daily, 1 weekly and 5 manual backups), capture one before risky changes (`heroku pg:backups:capture`), and keep pulled snapshots on an encrypted disk. [09 D1](09_STAGE2_3_PLAN.md#41-data) adds a freeze step that hashes each snapshot.
4. **P1 · Versions travel with every record.** *Code:* every record carries `schema_version` (1.2.0) and `app_version` (0.2.0). The session carries `protocol.protocol_version` (2.0.0), `protocol.seed`, the full plan, `consent.version`, `settings.review_mode` and `calibration.method`. Each trial carries `segmentation.method` (`online_v1`) and the resolved parameters. The rules:
   - bump the minor digit of `SCHEMA_VERSION` for additive changes;
   - `APP_VERSION` must equal the version in `package.json` (a unit test checks this);
   - bump `PROTOCOL_VERSION` whenever presets or ordering change;
   - tag the commit of every deployment used for collection.
5. **P1 · Version the offline steps.** Each derived file records the git commit, the normalisation version (`norm_v2`), the segmentation method and the version of the exclusion list.
6. **P1 · Provenance.** For each snapshot keep when it was pulled, with which filters, and its hash.
7. **P1 · Expect several copies of a record.** *Code:* the app re-sends the session record after every trial and saves an attempt more than once (a placeholder before the hand question, the decided copy, possibly a revoked copy). The service keeps the newest session (`updated_at`) and ranks trial copies placeholder < decided < revoked, as `motionlab.io` does ([Data schema §10](02_DATA_SCHEMA.md#10-deduplication-rules)). Attempts with `attempt` > 1 are separate recordings of the same planned trial.
8. **P1 · One person, two codes.** A second phone, another browser or the Home Screen copy starts a new participant code, and an undetected duplicate leaks across folds ([§8](#8-statistics)). Ask friends to stay in one browser on one phone. The admin page shows names, which makes duplicates easy to spot; merge them in the alias table of [09 D2](09_STAGE2_3_PLAN.md#41-data).
9. **P1 · Keep synthetic and real data apart.** *Code:* every record from `?sim=1` or `npm run synth` has `simulated: true`; synthetic study codes are `synthetic-v1` (the app's generator) and `synthetic-py` (`motionlab.synth`). Records made in simulation mode on the deployed app reach the same database: the admin page marks them, and `pull_live.py --real-only` leaves them out.
10. **P2 · One `study_code` per wave.** The street wave still uses `nose-touch-v1`; `protocol_version` tells it apart from the pilot. Give a distinct later wave (a validation cohort, a second round) its own code in `config.json`.
11. **P2 · Size.** The pilot's 23 trials took 1.6 MB as JSON and 1.0 MB in Postgres (JSONB is compressed). Street trials are larger, with the return of the arm and the extra streams: about 50–55 KB of JSON each in synthetic data, so about 1.5–2 MB per visit (29–39 trials). An Essential-0 database (1 GB) holds hundreds of visits.

## 8. Statistics

1. **P0 · The unit is the person, not the trial.** *Fact:* record-wise cross-validation overstates accuracy: across 47 papers, the median reported error was 5.6% with record-wise splits and 13.0% with subject-wise splits ([Saeb 2017][saeb-2017]). Identity confounding is common in digital-health ML ([Chaibub Neto 2019][chaibub-2019]), and the same arm movement works as a personal biometric ([Conti 2011][conti-2011]). Group every split by person (`person_key`, 09 D2), tune inside the training folds, and augment inside the estimator's `fit` so that mirrored and flipped copies never reach a validation fold ([scikit-learn][sklearn-groupkfold]; [09 D13](09_STAGE2_3_PLAN.md#44-models)).
2. **P0 · Lock the test set by rule.** [09 D3](09_STAGE2_3_PLAN.md#41-data): a person is in the locked test iff a hash of `person_key` is 0 mod 4 (about 25%); the student and the pilot are always excluded; a separate script opens it once and logs `locked_test_opened_at`.
3. **P0 · Report uncertainty at the person level.** Use a person bootstrap with B ≥ 2,000 (the implemented bootstrap resamples people, one level; [Saravanan 2020][saravanan-2020]). *Fact:* with about 100 test samples, CV accuracy has error bars of about ±10%, and the fold standard error underestimates them ([Varoquaux 2018][varoquaux-2018]). Also show the distribution of per-person accuracy (median, worst decile).
4. **P1 · Use one model for subgroup effects.** Fit a GLMM on out-of-fold correctness: `correct ~ hand*handedness + sex + age + speed + orientation + platform + (1|participant)` ([lme4][bates-2015]). The number of errors limits the model size, at roughly 10 errors per fixed-effect degree of freedom ([Peduzzi 1996][peduzzi-1996]; criticised by [van Smeden 2016][vansmeden-2016]). Collapse sparse levels or penalise.
5. **P1 · Correct for multiple comparisons.** With grips × speeds × variants × subgroups, some spurious "findings" are practically certain. Pre-specify 1–3 primary hypotheses, for example forced-choice accuracy on unseen people and the `hand × handedness` interaction. Treat everything else as exploratory: report effect sizes with CIs, control the false-discovery rate (Benjamini–Hochberg), and run no per-cell tests on trials.
6. **P1 · Size the sample from the first data.** With the 28 instructed trials of a street visit (tiers 1–3) the design effect is 1 + 27·ICC; at ICC 0.15, 30 people give a CI half-width of about ±4.6 points at 90% accuracy ([01 §10](01_PROTOCOL.md#10-participants-recruitment-and-sample-size); [09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection)). Re-estimate the ICC after 10 people; simulate power with [simr][simr-2016] if needed.
7. **P1 · Pre-register.** The commit of [09](09_STAGE2_3_PLAN.md) fixes the primary metric, the default model family, the gates, the τ target and the locked-test rule (09 M0). Add the exclusion rules and the subgroup model before opening the test set.
8. **P2 · Define the metric.** Primary: forced-choice accuracy on unseen people. Secondary: accuracy per person, log loss, Brier score, ECE and coverage under abstention ([Stage 3 predictor](08_STAGE3_PREDICTOR.md)).
9. **P2 · The handedness add-on has one label per person.** *Fact:* an AUC of 0.75 has a 95% CI of about ±0.16 with 15 left-handers and 45 right-handers, and about ±0.12 with 30 and 30 (Hanley–McNeil, [1982][hanley-mcneil]). Report it as exploratory.
10. **P2 · Robustness splits** are secondary: leave-one-grip-, -platform-, -speed-, -variant- and -device-model-out.

## 9. Recruitment

1. **P0 · Friends, adults, both platforms.** The participants are the researcher's college friends, 18 or older, on their own phones, recruited in person (often on the street) or with a WhatsApp link ([Field guide](10_FIELD_GUIDE.md)). One session per person, on one phone, in one browser.
2. **P0 · Ask left-handed friends on purpose.** *Fact:* prevalence is 10.6%, or 9.3–18.1% depending on the criterion ([Papadatou-Pastou 2020][papadatou-2020]), so 40 random friends give about 4. Aim for at least 8, ideally 10 or more ([09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection)).
3. **P1 · Both platforms:** iPhone users at least 25%, ideally 30% or more. The admin page shows each participant's platform.
4. **P1 · Balance sex within handedness groups.** Left-handedness is slightly more common in men, with an odds ratio of 1.23 ([Papadatou-Pastou 2008][papadatou-2008]).
5. **P1 · "Both" as the dominant hand** (`ambidextrous`) forms its own group. EHI classes exist only in protocol-1.0.0 data ([Veale 2014][veale-ehisf]).
6. **P1 · Age spread.** College friends cluster in their twenties; state it as a limitation, or widen the circle if age effects matter to the PI.
7. **P1 · Phones only.** Tablets differ in mass and grip.
8. **P2 · Device variety:** two or three Android brands at least, and many models.
9. **P2 · Keep a recruitment table** of counts by handedness × platform × sex. The admin page's participant table (name, age, sex, handedness, phone hand, platform, left/right trial counts) is its source.

## 10. Reproducibility

1. *Code:* exact dependency versions are pinned in `app/package.json`; CI and deploy install them with `npm ci` from `package-lock.json`. `analysis/requirements.txt` gives minimum Python package versions; freeze the environment used for reported results (`pip freeze > analysis/requirements.lock.txt`).
2. *Code:* the session plan is deterministic for a given preset, seed, posture, first hand and custom spec (`buildPlan`), and each session stores its seed and the full plan.
3. *Code:* `online_v1` is specified exactly in `app/src/core/segmentation.ts`, and the Python port (`analysis/motionlab/segment.py`, `compare_with_stored`) reproduces the stored indices; golden fixtures pin it. Keep that check passing.
4. Synthetic data are reproducible from their seed (`npm run synth -- --preset street --seed 42 …`). Use them for regression tests and to show the pipeline to the PI.
5. *Code:* CI runs the app typecheck and unit tests plus the Python tests (`analysis/tests`, `server`) on pushes to `main` and on pull requests (`.github/workflows/ci.yml`); the Pages deploy runs the unit tests before building. The Playwright journeys run locally, headless and silent, and never reach the live backend (`npm run e2e:live`, the smoke test of the deployed site, is the exception); the upload service has its own test suite.
6. Keep a dated change log of every change to the protocol, app, backend and `config.json` during collection. It decides which sessions are comparable.
7. Produce every figure and table with a script from a named snapshot and commit, with no manual edits ([09 D31](09_STAGE2_3_PLAN.md#6-report-and-deck)).
8. For the write-up, describe in the methods: the browser rate cap and quantisation; the implicit calibration and the sign and axis normalisation; the segmentation method and its parameters; the hands-free acceptance rule; the exclusion rules; the CV scheme and the bootstrap.

## 11. What could go wrong in front of the PI

| What happens | Why | Prevent | Recover live |
|---|---|---|---|
| The sensors do not start, or permission is denied | iOS remembers a Cancel until Safari is fully closed. Other causes: an in-app browser, Brave, or a tunnel URL that has expired. | Give the PI the Pages URL (`?predict=1`), never a tunnel. Test that exact URL on an iPhone and an Android phone the day before. | Follow the steps on screen. iOS in Safari: close Safari in the app switcher and reopen the link; a Home Screen app: close it and reopen it. Never switch to a Private tab. Android: site settings → Motion sensors. Otherwise switch to your own test phones. |
| No beep | Volume down, Bluetooth headphones, or audio not unlocked yet (iOS plays sound only after a tap). | Check the volume and Bluetooth. | The colour cues still work, and the screen always faces the PI. A laptop running the simulator is silent by design. |
| Blank page or an old version | A deploy on demo day, or a stale service worker. | No pushes to `main` in the 48 h before the demo. Check the live version the evening before. | Close and reopen the app, then reload. |
| Wrong prediction on the first try | Even at 95% per-trial accuracy, the chance of at least one error in 10 trials is 1 − 0.95¹⁰ ≈ 40%. | Agree on a series beforehand (12 attempts per person, [09 D29](09_STAGE2_3_PLAN.md#59-the-pi-evaluation)) and the expectation with its CI. | Show the running tally. Do not argue about single trials. |
| The PI moves differently from the training data (midline start, screen facing away, very slow, seated, a case) | The model knows only what was collected; screen-away grips exist only as synthetic training data. | Know your weak conditions from the leave-one-out tables; rehearse with 3–5 new people who include these conditions (09 D29). | Report per-condition accuracy honestly. Show "not sure" as a valid answer. |
| "No movement detected" or "That took too long" | The PI does not hold still before the beep or at the nose. | Demonstrate the rhythm once: hold still, beep, move, hold, double beep. | Repeat the attempt (a retry is not scored). |
| The PI's platform is under-represented in the data | Sign and quantisation differ between platforms. | Have leave-one-platform-out results ready; ask the PI's phone early. | The 2.5-s upright check refuses when the sign disagrees with the engine. |
| Wi-Fi fails | — | Predict mode makes no network requests at all ([09 D28](09_STAGE2_3_PLAN.md#57-network-and-privacy)). | Continue offline. |
| "Does the model see the label?" | A fair question. | *Code:* `Predictor.predict` uses only the sensor streams, never `trial.condition` (`app/src/core/predictor.ts`); a test randomises the condition. Point to the person-wise CV and the locked test set. | Show the code path. |
| "Why only 60 Hz?" | Browsers cap the rate. | Have the arguments of §6 ready. | Show the sources. |
| The phone dies or throttles | Low battery, or iOS Low Power Mode. | Charge the phone and turn Low Power Mode off. | Use a spare phone. |
| The screen rotates in sideways attempts | iOS cannot lock the screen orientation from a web page ([MDN data][bcd-screenorientation]). | Turn on Rotation Lock on iOS, or turn off auto-rotate on Android. | Repeat the attempt. |

## 12. Open questions for the PI

1. **Reuse:** may the data be reused beyond the admission project, for example in a publication? (Collection from adult friends goes without an ethics review: the researcher's decision.)
2. **Hosting:** is a Heroku Postgres database acceptable (its region: `heroku apps:info`), or must the data stay elsewhere?
3. **Retention and access:** how long are the raw data kept, and who may access them?
4. **Population:** Israel only (no GDPR)? Friends only, or a wider circle later?
5. **Canonical movement:** where does it start (in front of the same shoulder or at the midline, and at what height)? Is "any part of the phone touches the nose" acceptable? Standing or sitting?
6. **Evaluation:** the questions of [09 §10](09_STAGE2_3_PLAN.md#10-open-questions-for-the-pi): the PI's phone, forced choice or abstention, people and attempts, display, the 2.5-s upright check, per-user calibration, odd movements.
7. **Success criterion:** what target accuracy, measured how, and on which conditions?
8. **Sampling rate:** is the browser's 60 Hz acceptable, or is a native high-rate arm required (Apple Developer Program at US$99 per year, plus TestFlight review)?
9. **Sample size and timeline:** are the targets of [09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection) (30–40+ people, 8–10+ left-handers, 25–30% iPhone) right, and what is the deadline?
10. **Heading streams:** should the app stop storing the compass, absolute orientation and magnetometer altogether?
11. **Handedness add-on:** part of the evaluation, or a later extra?
12. **Odd movements:** robustness conditions only, or a separate "odd" class?
13. **Pre-registration:** is the dated commit of 09 enough, or should it also go to OSF?

[a13-translation]: https://ebnlaw.co.il/wp-content/uploads/2024/09/Amendment-13-Israeli-Privacy-Law-non-formal-translation.docx
[iapp-a13]: https://iapp.org/news/a/israel-marks-a-new-era-in-privacy-law-amendment-13-ushers-in-sweeping-reform
[pearl-a13]: https://www.pearlcohen.com/the-knesset-enacted-a-key-amendment-to-the-israeli-protection-of-privacy-law-summery-version/
[loc-a13]: https://www.loc.gov/item/global-legal-monitor/2025-11-17/israel-amendment-to-privacy-protection-law-goes-into-effect/
[conti-2011]: https://iris.unitn.it/handle/11572/89854
[gdpr]: https://eur-lex.europa.eu/eli/reg/2016/679/oj
[edpb-3-2018]: https://www.insideprivacy.com/data-privacy/edpb-adopts-final-version-of-guidelines-on-territorial-scope-of-the-gdpr/
[gh-pages-about]: https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages
[ploe-2019]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6538999/
[papadatou-2020]: https://research-repository.st-andrews.ac.uk/handle/10023/19889
[papadatou-2008]: https://wrap.warwick.ac.uk/29434/
[sainburg-2002]: https://link.springer.com/doi/10.1007/s00221-001-0913-8
[przybyla-2012]: https://pure.psu.edu/en/publications/dynamic-dominance-varies-with-handedness-reduced-interlimb-asymme/
[minjerk-smooth]: https://www.socsci.ru.nl/wilberth/physics/smooth.html
[stisen-2015]: https://portal.findresearcher.sdu.dk/da/publications/smart-devices-are-different-assessing-and-mitigating-mobile-sensi/
[a20-tremor]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9547012/
[cr-round]: https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h
[wk-motion-mm]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm
[wk-motion-h]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.h
[chrome-151]: https://developer.chrome.com/release-notes/151
[safari-26-notes]: https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes
[cr-linacc-fusion]: https://raw.githubusercontent.com/chromium/chromium/main/services/device/generic_sensor/linear_acceleration_fusion_algorithm_using_accelerometer.cc
[cr-pump]: https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_sensor_event_pump.h
[cr-traits]: https://github.com/chromium/chromium/blob/main/services/device/public/cpp/generic_sensor/sensor_traits.h
[phyphox-rates]: https://phyphox.org/forums/printthread.php?tid=2356
[w3c-2014-sign]: https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html
[gaitscope-96]: https://github.com/soroushdty/gaitscope/issues/96
[w3c-do]: https://w3c.github.io/deviceorientation/
[w3c-pr43]: https://github.com/w3c/deviceorientation/pull/43
[wk-dme-cpp]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/DeviceMotionEvent.cpp
[cr-dm-pump-cc]: https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc
[ff-nsdevicesensors]: https://github.com/mozilla-firefox/firefox/blob/main/dom/system/nsDeviceSensors.cpp
[wk-permission]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/dom/DeviceOrientationAndMotionAccessController.cpp
[wk-permission-cache]: https://github.com/WebKit/WebKit/blob/main/Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp
[webkit-itp-7day]: https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/
[brave-sensors]: https://community.brave.app/t/device-motion-data-accelerometer-gyro-not-available-on-android/292620
[apple-dts-100hz]: https://developer.apple.com/forums/thread/813136
[android-sensors]: https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview
[expo-go-may-2026]: https://expo.dev/changelog/expo-go-loading-changes-may-2026
[saeb-2017]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/
[chaibub-2019]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6789029/
[sklearn-groupkfold]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
[saravanan-2020]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290
[varoquaux-2018]: https://arxiv.org/abs/1706.07581
[bates-2015]: https://www.jstatsoft.org/article/view/v067i01
[peduzzi-1996]: https://doi.org/10.1016/S0895-4356(96)00236-3
[vansmeden-2016]: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-016-0267-3
[simr-2016]: https://doi.org/10.1111/2041-210X.12504
[hanley-mcneil]: https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747
[veale-ehisf]: https://www.millisecond.com/library/v7/edinburghhandednessinventory/edinburghhandednessinventory_sf.manual
[bcd-screenorientation]: https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json
