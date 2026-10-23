# 06 · Considerations and risk checklist

This checklist covers what can invalidate the data, the analysis or the PI's evaluation even when the code works. Read it before the pilot, then again before the main collection.

**Related:** [Project overview](00_PROJECT_OVERVIEW.md) · [Protocol](01_PROTOCOL.md) · [Data schema](02_DATA_SCHEMA.md) · [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md) · [Segmentation](04_SEGMENTATION.md) · [Modeling plan](05_MODELING_PLAN.md) · [Deployment](07_DEPLOYMENT.md) · [Stage 3 predictor](08_STAGE3_PREDICTOR.md) · data contract [`schema/records.schema.json`](../schema/records.schema.json)

**Tags**

| Tag | Meaning |
|---|---|
| **P0** | Blocks collection from other people, or silently invalidates data. Do it before the pilot. |
| **P1** | Do it before the main collection, after the pilot. |
| **P2** | Good practice. Do it when time allows. |
| *Fact* | A platform, legal or literature statement, with a source. |
| *Code* | Behaviour of this repository. The file is named. |
| (no tag) | Our recommendation. |

## 0. Critical path (every P0 item)

1. Get ethics approval from the right Bar-Ilan committee before anyone except you records data ([§1](#1-ethics-and-law)).
2. Finish the consent text, set `consent_version`, keep `min_age: 18` ([§1](#1-ethics-and-law)).
3. Treat the upload endpoint as public: schema validation, a size cap, no storage of rejected uploads (only a capped, identifier-only quarantine log), a private Drive folder ([§2](#2-privacy-and-security)).
4. Show the safety wording and repeat it: firm grip, gentle touch, stop on pain ([§3](#3-participant-safety-and-comfort)).
5. Flag wrong-hand and wrong-orientation trials. Never drop them silently and never keep them silently ([§4](#4-label-quality)).
6. Record both hands for every participant, and recruit left-handers on purpose ([§5](#5-confounds), [§9](#9-recruitment)).
7. Take the iOS sign inversion and the `rotationRate` axis mapping from the calibration. Never assume them ([§6](#6-sensor-and-platform-pitfalls)).
8. Never commit raw data, never edit it, and back it up every week ([§7](#7-data-management)).
9. Evaluate by participant: subject-wise CV, a locked test set and participant-level CIs ([§8](#8-statistics)).
10. Rehearse the demo risks on the kind of phone the PI is likely to use ([§11](#11-what-could-go-wrong-in-front-of-the-pi)).

## 1. Ethics and law

1. **P0 · Get approval first.** *Fact:* Israeli universities require ethics-committee approval before any contact with human participants. At Bar-Ilan, faculty and department committees review requests on their own forms ([Bar-Ilan research authority][biu-ethics]; [example form][biu-edu-form]). Ask the PI which committee covers the lab and whether an existing lab protocol can be amended ([Q1](#12-open-questions-for-the-pi)). Until approval, record only yourself or the simulator (`?sim=1`), and keep `upload.mode: "none"` in `app/public/config.json`.

2. **P0 · Write the full consent text.** *Code:* the app shows a consent screen and stores `consent.version` and `consent.accepted_at` in every `SessionRecord` (`app/src/app/sessionFlow.ts`). Check the text against the 45 CFR 46.116 elements ([eCFR][cfr-46-116]). It must also meet the notice duty in section 11 of the Privacy Protection Law: the purpose, whether answering is mandatory, and who receives the data ([Amendment 13, unofficial translation][a13-translation]).

   | Element | What to state for NoseTouch |
   |---|---|
   | Purpose | Whether phone motion sensors can tell which hand brought the phone to the nose. Lab admission project under [PI]. |
   | Procedure, duration | Profile questions, sensor check, short calibration, then the session: `quick` 24 trials (≈ 7.5 min), `standard` 56 (≈ 14 min) or `full` 132 (≈ 29.5 min). *Code:* `estimateMinutes` in `app/src/core/protocol.ts`; these estimates leave out the first-time setup and redos. The in-app consent says about 10–30 minutes, and about 15–20 minutes for the standard session including the setup (`duration` in `app/src/i18n/en/onboarding.ts`). Keep the approved text and the app's text the same. |
   | Risks | Minimal: bumping the nose, face or glasses, dropping the phone, arm fatigue. |
   | Benefits | None directly. |
   | Voluntary, withdrawal | Pause or end at any time without penalty. Deletion is requested with the participant code. |
   | Data collected | Motion and orientation streams and screen touches during trials, device and browser details, and profile answers. Required: age, sex, handedness and the hand usually used for the phone (sex, handedness and phone hand offer neutral answers). Optional: the EHI-SF, height, phone model, phone case and arm condition. *Code:* `validateDraft` in `app/src/app/screens/ProfileScreen.tsx`. |
   | Data not collected | Name (the nickname stays on the phone), camera, microphone, location. |
   | Technical data seen by providers | GitHub (site host) and Google (upload backend) process IP addresses ([GitHub Pages][gh-pages-about]). |
   | Pseudonymisation | A random code `P-XXXXXX`. No key on the server links codes to names. |
   | Storage, access | Where the data are stored (Drive of the study's Google account, or the lab laptop) and who can access them (student, PI). |
   | Retention | Until [date or N years], then deleted. The PI decides. |
   | Use | The admission project. Any further use (publication, lab reuse, open data) only if it is stated here. |
   | Eligibility | Aged 18 or over, located in Israel, no arm or hand condition that makes the task painful. |
   | Contact | `contact.name` and `contact.email` from `config.json`. |

3. **P0 · Version the consent text.** Bump `consent_version` in `config.json` (for example `2026-11-v2`) whenever the text changes. *Code:* the value is copied into each session, so you can later tell which text each participant accepted.

4. **P0 · Adults only.** *Code:* `min_age: 18` in `config.json` gates the profile. Age is self-reported, so the gate filters but does not verify. Minors need parental consent, so keep them out unless the committee decides otherwise.

5. **P1 · Amendment 13 to the Privacy Protection Law.** *Fact:* it has been in force since 14 Aug 2025. It broadens "personal information" to include online identifiers and adds a category, "information of special sensitivity", that includes biometric identifiers used for computerised identification. It also strengthens enforcement and narrows database registration ([IAPP][iapp-a13]; [Pearl Cohen][pearl-a13]; [Library of Congress][loc-a13]). *Fact:* the law defines "biometric" as "a unique human, physiological or behavioral characteristic that can be computationally measured" ([translation][a13-translation]), and arm movements are individual enough to serve as a biometric ([Conti et al. 2011][conti-2011]). So:
   - state in the consent and the data plan that motion data are not used to identify people;
   - minimise: age instead of birth date, no names, no location;
   - ask the Bar-Ilan privacy officer whether raw motion recordings fall under the biometric definitions.

   Registration (new s.8a(a)(1)) is required only when the database's main purpose is collecting personal information to pass to others as a business or for payment and it holds data on more than 10,000 people, or when the controller is a public body. Separately, a database holding special-sensitivity information on more than 100,000 people must notify the Authority. Some secondary summaries conflate the two thresholds. A study of a few dozen pseudonymous participants is far below both; ask the privacy officer whether the university counts as a public body for this purpose.

6. **P1 · GDPR: restrict participation to people located in Israel.** *Fact:* the GDPR applies to non-EU controllers that monitor the behaviour of people in the EU (Art. 3(2)(b); [GDPR][gdpr]). The EDPB counts monitoring through smart devices followed by behavioural analysis ([EDPB 3/2018 summary][edpb-3-2018]). Pseudonymised data remain personal data for the controller (Recital 26). Write "Israel only" in the consent. If the study needs EU participants, add a GDPR privacy notice, explicit consent, Art. 89 safeguards and hosting in Israel or the EU.

7. **P1 · Data location.** Uploads land in the Drive of a Google account ([§2](#2-privacy-and-security)). Ask the committee whether that is acceptable or whether the data must stay on Bar-Ilan infrastructure ([Q2](#12-open-questions-for-the-pi)). The local receiver (`server/receiver.py`) keeps lab-session data on a lab laptop ([Deployment §5](07_DEPLOYMENT.md#5-local-receiver-for-lab-sessions)).

8. **P1 · Retention and deletion.** Agree a retention period with the PI. When a participant sends their code to ask for deletion:
   1. delete the folder `<study_code>/<participant_id>/` in Drive, for every `study_code` the participant took part in;
   2. delete their rows in the `sessions`, `trials` and `quarantine` sheets of the index spreadsheet;
   3. delete them from every local snapshot and derived dataset; for lab sessions also `server/received/<study_code>/<participant_id>/` and the participant's lines in `server/received/quarantine.log`;
   4. log the date and the code, and nothing else.

   *Code:* the backend never stores a rejected upload as a file, and its `quarantine` sheet holds only metadata and identifiers, so steps 1 and 2 cover everything on the server. Older script versions did keep rejected payloads in a `quarantine/` Drive folder; if one exists, delete it ([backend README](../backend/google-apps-script/README.md)). Deleting inside the app (`LocalStore.deleteParticipant`, `clearAll`) removes only the copies on that phone, and the consent text must say so. Tell participants to note their code.

9. **P2 · Future use.** Decide now whether the data may be reused: publication, later lab projects, open data. Consent is hard to extend afterwards.

## 2. Privacy and security

1. **P0 · Use pseudonymous identifiers only.** *Code:* `participant_id` is a random `P-XXXXXX` code over a 32-symbol alphabet without 0, O, 1 and I (schema pattern `^P-[A-HJ-NP-Z2-9]{6}$`, about 1.07 × 10⁹ codes). Never ask for names, e-mail addresses or phone numbers in the app. Keep any sign-up sheet for lab sessions offline and separate, and never join it to the data.

2. **P0 · The nickname stays on the phone.** *Code:* `nickname` is replaced by `null` before upload unless `upload_nickname: true` (`redactProfile` and `redactSession` in `app/src/core/config.ts`). Keep it `false`. The nickname only tells profiles apart on a shared phone.

3. **P0 · The upload endpoint is public.** *Fact:* `config.json`, which holds the endpoint and `study_key`, is a public file. A GitHub Pages site is public even when its repository is private ([GitHub][gh-pages-about]). *Code:* `study_key` is "a spam filter, not real security" (`app/src/core/types.ts`). The backend must meet these requirements (check them in `backend/google-apps-script/`):
   - It validates every envelope against `schema/records.schema.json`: at least `kind`, `record_type`, the id patterns and `schema_version`.
   - It caps the body size. Google documents no maximum POST size for web apps ([Apps Script web apps][gas-web]), so set one yourself: about 2 MB, since a trial is typically tens of KB.
   - It never stores a rejected payload (malformed, oversized, wrong study key) and never evaluates its content. *Code:* it logs one row of metadata (reason, size, hash, record identifiers) in the `quarantine` sheet, at most 500 rows per UTC day, so junk posts cannot fill Drive or the Sheet.
   - It answers with a short status object only, never with stored data.

   If spam appears, rotate the key: set a new `STUDY_KEY` in the Script Properties and the same value as `upload.study_key` in `config.json`. Phones read `config.json` when the app starts, so a phone that is still open keeps the old key until the app is reloaded. Until then the backend rejects its uploads (`bad_study_key`) without storing them. *Code:* the app keeps those records, does not retry a permanent rejection (`retry: false`) on its own, and sends them again when someone taps "Upload now" after the reload. Rotate between collection days, and ask anyone with the app open to close and reopen it.

4. **P0 · Use a private storage account.** Run the backend from a dedicated Google account with 2-step verification. The script needs the full Drive scope ([DriveApp][driveapp]), so a dedicated account limits what a leak can expose. Never share the folder with "anyone with the link". Add the PI by named account. Only people who may read all the data may edit the script, because "Execute as: Me" runs every request with the owner's access.

5. **P1 · No location, camera or microphone, and drop the heading.** *Code:* the app asks only for motion and orientation access (`app/src/core/sensors.ts`), and the beeps need no microphone. The orientation stream does keep a `compass` column (iOS `webkitCompassHeading`) as metadata. A heading is not a location, but it partly identifies the place (the room's orientation) and carries no left/right information. Exclude it from features and from any dataset shared outside the lab ([Q10](#12-open-questions-for-the-pi)).

6. **P1 · Device details are personal data too.** The user agent, `model_hint`, screen size, language and memory partly identify a person. Amendment 13 counts online identifiers as personal information. Keep them for device analyses. Coarsen them to platform, browser and major version before sharing outside the lab.

7. **P2 · Transport and repository hygiene.** *Code:* `config.ts` accepts only `https://` endpoints or same-origin paths. `.gitignore` excludes `data/`, `analysis/data/*`, `analysis/reports/*`, `server/received/` and `.env*`. Keep account-specific ids (Drive folder, Sheet) out of the public repository.

8. **P1 · Treat downloaded records as untrusted text.** Anyone who finds the endpoint and the public study key can post a schema-valid but fabricated record, and free-text fields such as `quality.warnings`, `outcome.note`, `phone_model` and `session.notes` can then carry spreadsheet formulas. *Code:* the backend writes Sheet cells as plain text, and `qa_report.py` and `train_baseline.py` formula-escape the researcher-facing CSVs they write (`motionlab.report.csv_safe`): a text cell that starts with a formula character such as `=`, `+`, `-` or `@` gets a leading `'`. Still filter every analysis to the intended `study_code` and `simulated == false`, and import other CSV exports as text rather than opening them directly.

## 3. Participant safety and comfort

1. **P0 · Firm grip, gentle touch.** *Code:* the app shows "Hold the phone firmly (doing this over a bed or sofa is a good idea) and always touch your nose gently", and the fast-speed instruction ends with "Touch your nose gently!" (`app/src/i18n/en/common.ts`). Repeat this aloud in lab sessions: "fast" means a fast reach that ends softly.
2. **P0 · Stop on pain.** Anyone in pain stops, with no pressure to finish. Analyse separately the participants who answer yes to the optional `arm_issue` question.
3. **P1 · Glasses.** A fast reach can hit the frame. Suggest removing glasses or approaching gently, and note it in the lab log.
4. **P1 · Fatigue.** Holding a phone of about 0.2 kg at arm's length tires the arm. *Code:* the holds are short: 600 ms of stillness, then a random 0.5–1.2 s foreperiod (`DEFAULT_SEGMENTATION_PARAMS`), and a rest screen appears between blocks. Prefer `standard` (≈ 14 min) for remote participants, and run `full` (≈ 29.5 min) in the lab with breaks. *Code:* an interrupted session can be resumed within 24 h (`findResumableSession`).
5. **P1 · Space and posture.** Use a clear space and stand still or sit (posture is recorded per session). Never record while walking, driving or on stairs.
6. **P2 · Odd variants.** `lean_in`, `elbow_high`, `chin`, `forehead` and `jerky` are unfamiliar movements. *Code:* the odd block opens with a warning that every movement is different, each trial shows its instruction, and participants can tap Redo or flag a Problem.
7. **P2 · Hearing and language.** *Code:* every cue is both a sound and a large colour change, and the UI is in English and Hebrew (`?lang=he`, right-to-left). A participant who cannot hear the beeps sees the colour cues only when the screen faces them. In `*_away` orientations they depend on the audio.

## 4. Label quality

The label is the instructed hand and orientation. *Fact:* in a supervised 7-day wrist-sensor study, 7 of 45 participants (15.6%) wore a sensor on the wrong wrist on at least one day ([PLOE, Sensors 2019][ploe-2019]). Expect 5–15% protocol deviations in a self-administered app.

| Problem | Prevention and detection in the app (*Code*) | Offline check |
|---|---|---|
| Wrong hand | One hand per block, a "Switch hands!" screen, and the hand named on every Ready screen. The review step offers **Problem → I used the wrong hand** (`outcome.participant_flag = wrong_hand`). | All trials in a block use the same hand. Flag single trials whose prediction disagrees with the rest of their block. Flag trials where a mirror-consistent model is confident (p > 0.95) against the label. |
| Wrong orientation | `detected.rotation_start` and `rotation_end` (from the gravity vector) are compared with the instructed rotation in `detected.orientation_match`. On a mismatch the review step asks the participant to flag `wrong_orientation`. | Gravity cannot tell screen toward from away in upright and upside-down poses (`app/src/core/orientationDetect.ts`). Heuristic: in `*_toward` trials the nose often touches the screen and produces touch events, and in `*_away` trials it cannot. |
| Moving before GO | GO comes only after 600 ms of stillness plus a random 0.5–1.2 s foreperiod. An early movement gets an error beep and is counted in `timing.false_starts`. | Exclude trials with a short or noisy baseline. |
| No real movement, or no hold at the nose | The trial ends as `no_movement` or `too_long`, and the review shows a small plot of ‖ω‖ and acceleration. | Re-segment offline. Flag disagreements over 100 ms and movements with several peaks ([Segmentation](04_SEGMENTATION.md)). |
| Speed not as instructed | Instruction only. | Use the measured duration as the analysis variable, not the instruction. |
| Start-position drift (bent arm, centre instead of shoulder) | Instruction only. | Track the baseline gravity direction and the movement amplitude per participant across `trial_index`. |
| Odd variant not performed | Instruction only. | Count movement units (peaks of ‖ω‖). |
| Interrupted recording | Status `interrupted` when the page is hidden, plus `quality.hidden_during`, `quality.gaps_over_50ms` and `quality.motion_hz`. | Exclude trials with any gap over 50 ms or a mean rate under 50 Hz (recommendation from the sensor research, see [03](03_SENSORS_AND_PLATFORMS.md)). |
| Redo semantics | Every attempt is stored with `attempt` and `outcome.accepted`. Failed attempts are "kept for quality control but do not count". | Decide in the pre-registration whether non-accepted attempts are ever used. |
| Simulated data | `simulated: true` is set on every record made with `?sim=1` (never remembered across reloads) or with `npm run synth`. | Assert `simulated == false` in every analysis of real data. |

Watch for protocol drift. Plot each participant's duration, peak ‖ω‖ and false starts against `trial_index`, and compare the first and last blocks. Write the exclusion rules into the pre-registration before you see any model result.

## 5. Confounds

| Confound | Why it matters | Handled by the design (*Code*) | Still to do |
|---|---|---|---|
| Hand dominance | *Fact:* about 10.6% of people are left-handed ([Papadatou-Pastou 2020][papadatou-2020]), so for most participants "right hand" also means "dominant hand". Dominant and non-dominant reaches differ in curvature and initial direction ([Sainburg 2002][sainburg-2002]), and less so in left-handers ([Przybyla 2012][przybyla-2012]). The model can learn dominance instead of side. | Both hands for everyone, EHI-SF in the profile, mirror augmentation planned ([Modeling plan](05_MODELING_PLAN.md)). | At least 20–30 left-handers. Model `hand × handedness`. Report left-handers' accuracy separately. |
| Start pose | The start (in front of the shoulder or at the midline) sets how large the lateral component is. It is likely a main driver of errors. | `start_position` factor (`natural`, `shoulder`, `center`). | Report accuracy per start position. Agree the canonical pose with the PI ([Q5](#12-open-questions-for-the-pi)). |
| Instructed vs achieved speed | People read "fast" differently. Threshold segmentation bias grows with duration: with a minimum-jerk profile, a 5%-of-peak threshold cuts 5.9% of the duration off each end, 18 ms at 0.3 s but 119 ms at 2 s ([minimum-jerk model][minjerk-smooth]). | Per-speed segmentation overrides (`resolveParams`). Raw streams kept. | Analyse by measured duration. |
| Order, fatigue, learning | Behaviour changes with time in the session. | Hands alternate ABBA over orientation pairs, the first hand follows the seed's parity, and trials are shuffled within blocks. | `upright_toward` always comes first (familiarisation) and odd blocks always follow the main blocks, so both are confounded with time in the session. Interpret them with that in mind, and check the realised split of `protocol.first_hand`. |
| Device model, mass, case | *Fact:* differences between devices significantly degrade motion classifiers ([Stisen 2015][stisen-2015]). Phone mass shifts the tremor spectrum ([phone vs wearable][a20-tremor]). | `device.model_hint` (the model on Android, only the device class on iOS), `phone_model`, `phone_case`. | Leave-one-device-model-out. Clean up model names by hand, since iOS does not expose the model. |
| Platform quantisation and sign | *Fact:* Chrome rounds values to 0.1 m/s² and 0.1 °/s, WebKit does not ([Chromium][cr-round]). iOS reports acceleration with an inverted sign ([WebKit][wk-motion-mm]). | `calibration.gravity_sign` measured per session, raw values kept, and the within-person design keeps platform balanced across labels. | Quantise iOS data in preprocessing (or test both ways). Leave-one-platform-out. At least 30% of participants on each platform. |
| OS and browser updates during the study | *Fact:* Chrome 151 added `requestPermission()` and plans to make sensor access ask by default ([Chrome 151][chrome-151]). iOS 26 freezes the OS version in the user agent ([Safari 26][safari-26-notes]). No WebKit bug tracks the sign inversion, so a fix could ship without notice. | Calibration and engine detection in every session. `app_version`, `device.browser` and `device.os_version` stored. | Run QA by date every week and watch `gravity_sign`, rate and quantisation. Treat `os_version` as unreliable on iOS 26 and later. |
| Posture | Standing or sitting is set per session, so it is confounded with the person. | `posture` stored. | Use it as a covariate. Ask everyone for the same posture unless posture is a factor. |
| Setting (lab vs home) | Supervision changes behaviour. | n/a | Give each wave or setting its own `study_code` (editable in `config.json`). |
| Browser "acceleration" source | *Fact:* Chrome falls back to a crude low-pass gravity estimate when the phone has no linear-acceleration sensor ([Chromium fusion][cr-linacc-fusion]). | `segmentation.acc_source` stored. | Compute gravity and linear acceleration offline from `accg_*` and `rr_*`. |

## 6. Sensor and platform pitfalls

This is a summary. Mechanisms, measurements and the full sources are in [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md).

| Pitfall | Consequence | How it is handled |
|---|---|---|
| About 60 Hz maximum in every browser: WebKit uses a 1/60 s timer ([WebKit][wk-motion-h]), Chromium a 60 Hz pump ([Chromium][cr-pump]), and the Generic Sensor API caps at 60 Hz ([Chromium][cr-traits]). | A web page cannot get a higher rate. | 60 Hz is adequate: 99% of the acceleration energy of movements lasting 0.25 s or more lies below about 10 Hz (minimum-jerk spectrum, [model][minjerk-smooth]). Validate with phyphox, which gets about 100 Hz on iPhone and up to 500 Hz on Android ([phyphox forum][phyphox-rates]). |
| iOS reports `acc_*` and `accg_*` with all axes inverted: face-up z ≈ −9.81 ([W3C list 2014][w3c-2014-sign]; [2026 measurement][gaitscope-96]). | On chiral features, iOS trials look like the other hand. | The calibration measures `gravity_sign` and the data are normalised offline. Never infer iOS from `requestPermission`, which Chrome 151 has too. |
| `rotationRate` alpha/beta/gamma are rates about device x/y/z ([W3C CRD 2025][w3c-do]; [PR #43][w3c-pr43]). MDN's alpha page still says z. | Gyro channels get mislabelled and the mirror is wrong. | A tilt-gesture axis check (`calibration.axis_check.verdict`). |
| `event.interval` is ≈ 0.0167 (seconds) on iOS, 16 on Chrome and 100 on Firefox ([WebKit][wk-dme-cpp]; [Chromium][cr-dm-pump-cc]; [Firefox][ff-nsdevicesensors]). | Wrong time steps. | Only per-sample times are used. `interval_raw` is metadata. |
| Euler angles hit gimbal lock at the upright start pose ([W3C][w3c-do]). | alpha and gamma jump. | The orientation stream is metadata only. |
| Chrome quantisation: 0.1 m/s², 0.1 °/s ([Chromium][cr-round]). | The model can use it to tell the platform apart (a shortcut). | Quantise iOS data and stratify by platform. |
| iOS permission needs a tap, a Cancel stays cached until Safari is quit, and insecure pages are denied silently ([WebKit][wk-permission]; [WebKit cache][wk-permission-cache]). | Sessions with no data. | An explainer screen before the system prompt. Recovery steps are in [Deployment §8](07_DEPLOYMENT.md#8-troubleshooting). |
| No events arrive while the page is hidden ([W3C][w3c-do]). | Gaps. | Trial status `interrupted`, and a screen wake lock. |
| Safari deletes a site's storage after 7 days of use without interaction. Home Screen apps are exempt ([WebKit][webkit-itp-7day]). | Unsent data are lost. | Upload after every trial, the export button, and the Home Screen app for participants who come back on another day. |
| In-app browsers and WebViews behave unpredictably, and Brave on Android blocks the sensors ([Brave][brave-sensors]). | Odd permission behaviour or no data. | `device.in_app_browser` detection, then a request to open the link in Safari or Chrome. |
| Fast trials can go past ±4 g on some phones and reach 500–700 °/s (minimum-jerk estimate; a phone with a ±4 g accelerometer: [Galaxy A20 study][a20-tremor]). | Clipping. | Flag it in QA. |
| Vibration contaminates the accelerometer. | An artefact at GO. | *Code:* the phone vibrates only on `block_start` (`app/src/core/cues.ts`). |

Higher rates need native code. iOS apps get at most 100 Hz ([Apple DTS][apple-dts-100hz]). Android apps get up to 200 Hz, or more with a special permission ([Android][android-sensors]). Since May 2026 Expo Go cannot run other people's projects ([Expo][expo-go-may-2026]), so a native arm needs store or TestFlight distribution. Decide on it only after the phyphox validation ([Q8](#12-open-questions-for-the-pi)).

## 7. Data management

1. **P0 · Never commit data.** *Code:* `.gitignore` excludes `data/`, `analysis/data/*` (except its README), `analysis/reports/*` and `server/received/`. Real data live in Drive and in dated local snapshots under `analysis/data/` ([Deployment §7](07_DEPLOYMENT.md#7-collect-the-data-and-run-the-analysis-kit)).
2. **P0 · Raw data are read-only.** Never edit downloaded JSON. Put corrections (relabels, exclusions) in a versioned table keyed by `trial_id`, with a reason and a date.
3. **P0 · Back up.** The study's Google account is a single point of failure. Each week, download a snapshot to an encrypted lab disk and keep one more offline copy. The Drive web UI zips folders in 2 GB parts ([Google][drive-zip]), or you can sync with Drive for desktop.
4. **P1 · Versions travel with every record.** *Code:* every record carries `schema_version` (1.1.0) and `app_version` (0.1.0). The session carries `protocol.protocol_version` (1.0.0), `protocol.seed`, the full plan and `consent.version`. Each trial carries `segmentation.method` (`online_v1`) and the resolved parameters. The rules:
   - bump the minor digit of `SCHEMA_VERSION` for additive changes;
   - `APP_VERSION` must equal the version in `package.json` (a unit test checks this);
   - bump `PROTOCOL_VERSION` whenever presets or ordering change;
   - tag the commit of every deployment used for collection.
5. **P1 · Version the offline steps.** Each derived file records the git commit, the normalisation version, the segmentation method (`online_v1` as stored, or `offline_v1` recomputed) and the version of the exclusion list.
6. **P1 · Provenance.** For each record, keep where it came from (Drive snapshot, receiver folder or participant export file) and when it was downloaded. The index Sheet is the inventory.
7. **P1 · Expect duplicates.** *Code:* uploads are idempotent because the backend overwrites by record id. The session record is re-sent after every trial, and an upload that timed out may already be stored. In the analysis, keep one session per `session_id` (the latest `updated_at`) and unique `trial_id`s. Attempts with `attempt` > 1 are separate recordings of the same planned trial. The full rules are in [Data schema §10](02_DATA_SCHEMA.md#10-deduplication-rules).
8. **P1 · One person, two codes.** Safari and the Home Screen copy of the app keep separate storage, and so does a second phone. The same person can therefore appear under two participant codes, and the data cannot reveal it. An undetected duplicate person leaks across subject-wise folds ([§8](#8-statistics)). Ask participants to stay in one browser on one phone. Before splitting folds, look for near-identical profiles with close timestamps.
9. **P1 · Keep synthetic and real data apart.** *Code:* every record from `?sim=1` or `npm run synth` has `simulated: true`, and `npm run synth` uses the study code `synthetic-v1` by default. Records made in simulation mode on the deployed app go to the same backend as real ones. Synthetic sets go to `analysis/data/synthetic/` and are regenerated rather than stored. Filter on `simulated` and `study_code` in every analysis of real data.
10. **P2 · One `study_code` per wave**, for example `nose-touch-pilot`, `nose-touch-v1` and `nose-touch-validation`. Change it in `config.json` between waves.
11. **P2 · Size.** Records take about 8 KB per second of recording: a `standard` session is about 2–2.5 MB, a `full` session about 5 MB, and 100 `standard` participants about 0.25 GB ([Data schema §7](02_DATA_SCHEMA.md#7-storage-layout)). The free 15 GB of a Google account is shared with Gmail and Photos ([Google One][googleone-15gb]), so storage is not a constraint. A Sheets cell holds at most 50,000 characters ([Google][drive-limits]), which is why raw data never go into the Sheet.

## 8. Statistics

1. **P0 · The unit is the person, not the trial.** *Fact:* record-wise cross-validation overstates accuracy: across 47 papers, the median reported error was 5.6% with record-wise splits and 13.0% with subject-wise splits ([Saeb 2017][saeb-2017]). Identity confounding is common in digital-health ML ([Chaibub Neto 2019][chaibub-2019]), and the same arm movement works as a personal biometric ([Conti 2011][conti-2011]). Use `GroupKFold`, `StratifiedGroupKFold` or leave-one-subject-out with group = participant, tune inside the training folds (nested), and keep mirrored copies in their source participant's fold ([scikit-learn][sklearn-groupkfold]; [Modeling plan](05_MODELING_PLAN.md)). Merge any participant codes that belong to one person before splitting ([§7](#7-data-management)).
2. **P0 · Lock a test set now.** Pick about 20% of participants at random, stratified by handedness and platform, and write down their ids before training any model. Leave them untouched until the final model.
3. **P0 · Report uncertainty at the participant level.** Use a hierarchical bootstrap: resample participants, then their trials, at least 2,000 times ([Saravanan 2020][saravanan-2020]). *Fact:* with about 100 test samples, CV accuracy has error bars of about ±10%, and the fold standard error underestimates them ([Varoquaux 2018][varoquaux-2018]). Also show the distribution of per-participant accuracy (median, worst decile).
4. **P1 · Use one model for subgroup effects.** Fit a GLMM on out-of-fold correctness: `correct ~ hand*handedness + sex + age + speed + orientation + platform + (1|participant)` ([lme4][bates-2015]). The number of errors limits the model size, at roughly 10 errors per fixed-effect degree of freedom ([Peduzzi 1996][peduzzi-1996]; criticised by [van Smeden 2016][vansmeden-2016]). For example, 2,400 trials at 5% error give 120 errors, enough for about 12 degrees of freedom. Collapse sparse levels or penalise.
5. **P1 · Correct for multiple comparisons.** With 10 orientations × 3 speeds × 10 variants × several subgroups, some spurious "findings" are practically certain. Pre-specify 1–3 primary hypotheses, for example accuracy on unseen participants and the `hand × handedness` interaction. Treat everything else as exploratory: report effect sizes with CIs, control the false-discovery rate (Benjamini–Hochberg), and run no per-cell tests on trials.
6. **P1 · Size the sample by simulation after the pilot** ([simr][simr-2016]). Rough numbers from the design research: with 40 trials per person the design effect 1 + 39·ICC is 4.9 at ICC 0.1. Telling 5% from 15% error needs about 141 effective trials per group, which is about 18 people per group at ICC 0.1 and about 31 at ICC 0.2.
7. **P1 · Pre-register** before opening the test set, on OSF or in a dated internal document. Include the hypotheses, the primary metric, the exclusion rules, the CV scheme, the test-set ids, the subgroup model and the mirror-augmentation policy.
8. **P2 · Define the metric.** Primary: accuracy per participant, averaged over participants, so each person counts once. Secondary: pooled trial accuracy, log-loss, and calibration of `p_right` ([Stage 3 predictor](08_STAGE3_PREDICTOR.md)).
9. **P2 · The handedness add-on has one label per person.** *Fact:* an AUC of 0.75 has a 95% CI of about ±0.16 with 15 left-handers and 45 right-handers, and about ±0.12 with 30 and 30 (Hanley–McNeil method, [1982][hanley-mcneil]). Report it as exploratory.
10. **P2 · Robustness splits** are secondary: leave-one-orientation-out, leave-one-platform-out and leave-one-device-model-out.

## 9. Recruitment

1. **P0 · Pilot first.** Run 8–10 people on both platforms, after approval, to estimate the error rate and the ICC and to find protocol problems.
2. **P0 · Recruit left-handers on purpose.** *Fact:* prevalence is 10.6%, or 9.3–18.1% depending on the criterion ([Papadatou-Pastou 2020][papadatou-2020]), so random recruitment needs about 190 people to get 20 left-handers. Use targeted calls. Aim for at least 20, ideally 30, with EHI-SF class `left`.
3. **P1 · Both platforms**, at least 30% each. Check the iOS/Android split in the index Sheet every week.
4. **P1 · Balance sex within handedness groups.** Left-handedness is slightly more common in men, with an odds ratio of 1.23 ([Papadatou-Pastou 2008][papadatou-2008]).
5. **P1 · Mixed-handers** (EHI-SF LQ between −60 and 60; [Veale 2014][veale-ehisf]) form their own group.
6. **P1 · Age spread** across the adult range. If age effects matter to the PI, set quotas per age band (for example 18–29, 30–49, 50+).
7. **P1 · Phones only.** Tablets differ in mass and grip, and the protocol and model target phones.
8. **P1 · Validation cohort.** Record 10–20 participants on both platforms with both the PWA and phyphox on the same phone ([phyphox][phyphox-download]; [Modeling plan](05_MODELING_PLAN.md)).
9. **P2 · Device variety.** Include many phone models, with and without cases. On iOS ask for the model, because the browser does not expose it.
10. **P2 · Keep a recruitment table** updated weekly with counts by handedness × platform × sex × age band. Put the stopping rule in the pre-registration.

## 10. Reproducibility

1. *Code:* exact dependency versions are pinned in `app/package.json`, and CI and deploy install them with `npm ci` from `package-lock.json`. `analysis/requirements.txt` gives minimum Python package versions, with the tested versions listed in its header. Freeze the environment used for reported results (`pip freeze > analysis/requirements.lock.txt`).
2. *Code:* the session plan is deterministic for a given preset, seed, posture, first hand and custom spec (`buildPlan`), and each session stores its seed and the full plan.
3. *Code:* `online_v1` is specified exactly in `app/src/core/segmentation.ts`. The Python port (`analysis/motionlab/segment.py`, `compare_with_stored`) must reproduce the stored indices. Keep that check passing.
4. Synthetic data are reproducible from their seed (`npm run synth -- --seed 42 …`). Use them for regression tests and to show the pipeline to the PI.
5. *Code:* CI runs the app typecheck and unit tests plus the Python tests (`analysis/tests`, `server`) on pushes to `main` and on pull requests (`.github/workflows/ci.yml`). The Pages deploy runs the unit tests before building.
6. Keep a dated change log of every change to the protocol, app, backend and `config.json` during collection. It decides which sessions are comparable.
7. Produce every figure and table with a script from a named snapshot and commit, with no manual edits.
8. For the write-up, describe these in the methods:
   - the browser rate cap and quantisation;
   - sign and axis normalisation, and the calibration;
   - the segmentation method and its parameters;
   - the exclusion rules;
   - the CV scheme and the bootstrap.

## 11. What could go wrong in front of the PI

| What happens | Why | Prevent | Recover live |
|---|---|---|---|
| The sensors do not start, or permission is denied | iOS remembers a Cancel until Safari is quit. Other causes: the link opened in an in-app browser, Brave, or a tunnel URL that has expired. | Give the PI the Pages URL, not a tunnel. Test that exact URL on an iPhone and an Android phone the day before. | Follow the recovery steps on screen. iOS in Safari: quit Safari from the app switcher and reopen the link; a Home Screen app: close it in the app switcher and reopen it. Never switch to a Private tab, which starts with empty storage. Android: site settings → Motion sensors. Otherwise switch to your own test phones. |
| No beep | Volume is down or Bluetooth headphones are connected. iOS's ringer switch should not matter: the app requests the `playback` audio session on Safari 16.4+ ([MDN data][bcd-audiosession]). | Check the volume and Bluetooth. | Colour cues still work while the screen faces the PI. |
| Blank page or an old version | A deploy on demo day, or a stale service worker. | No pushes to `main` in the 48 h before the demo. Check the live version the evening before. | Close and reopen the app, then reload. |
| Wrong prediction on the first try | Even at 95% per-trial accuracy, the chance of at least one error in 10 trials is 1 − 0.95¹⁰ ≈ 40%. | Agree on a series beforehand (for example 10–20 trials with both hands) and show the CI. | Show the running tally. Do not argue about single trials. |
| The PI moves differently from the training data (midline start, landscape with the screen away, curved, very slow, seated) | The model knows only what was collected. | Know your weak conditions from leave-one-orientation-out and the per-condition tables. Agree the canonical movement ([Q5](#12-open-questions-for-the-pi), [Q6](#12-open-questions-for-the-pi)). | Report per-condition accuracy honestly. Show a `p_right` near 0.5 as "unsure". |
| "No movement detected" or "That took too long" | The PI does not hold still before the beep or at the nose. | Demonstrate the rhythm once: hold still, beep, move, hold, double beep. | Redo the trial and show the live signals in the sensor sandbox. |
| The PI's platform is under-represented in the data | Sign and quantisation differ between platforms. | Have leave-one-platform-out results ready. Run the calibration on the PI's phone. | Repeat the calibration. |
| Wi-Fi fails | Uploads cannot go out. | Prediction runs in the browser ([Stage 3 predictor](08_STAGE3_PREDICTOR.md)), and recording works offline. | Continue offline. |
| "Does the model see the label?" | A fair question. | *Code:* `Predictor.predict` uses only the sensor streams and the segmentation, never `trial.condition` (`app/src/core/predictor.ts`). Also point to the subject-wise CV and the locked test set. | Show the code path. |
| "Why only 60 Hz?" | Browsers cap the rate. | Have the phyphox validation results ready ([§6](#6-sensor-and-platform-pitfalls)). | Show the sources. |
| The phone dies or throttles | Low battery, or iOS Low Power Mode (its effect on the motion rate is undocumented). | Charge the phone and turn Low Power Mode off. | Use a spare phone. |
| The screen rotates in landscape trials | iOS cannot lock the screen orientation from a web page ([MDN data][bcd-screenorientation]). | Turn on Rotation Lock on iOS, or turn off auto-rotate on Android. | Redo the trial. |

## 12. Open questions for the PI

1. **Ethics:** which Bar-Ilan committee covers this work? Can an existing lab protocol be amended? Does the admission-stage pilot need approval? May the data be reused for publication?
2. **Hosting:** is a dedicated Google account (Drive) acceptable, or must the data stay on Bar-Ilan infrastructure or in Israel?
3. **Retention and access:** how long are the raw data kept, and who may access them?
4. **Population:** Israel only (no GDPR)? What age range? Compensation or course credit? Lab sessions, remote sessions, or both?
5. **Canonical movement:** where does it start (in front of the same shoulder or at the midline, and at what height)? Which part of the phone touches the nose? Standing or sitting?
6. **Evaluation:** will the PI use their own phone, and which model and OS? Single trials or a series? Odd movements, unusual orientations or `free` grips? May the app run a 30-second calibration on the PI's phone before predicting?
7. **Success criterion:** what target accuracy, measured how (per trial or per person), and on which conditions?
8. **Sampling rate:** is the browser's 60 Hz cap acceptable, or is a native high-rate arm required (Apple Developer Program at US$99 per year, plus TestFlight review)? Does the lab have an Apple Developer organisation account?
9. **Sample size and timeline:** how many participants in total, how many left-handers, what platform split, and what deadline?
10. **Compass heading:** should the app stop storing `compass` altogether?
11. **Handedness add-on:** is it part of the evaluation, or a later extra?
12. **Odd movements:** robustness conditions only, or a separate "odd" class?
13. **Pre-registration:** public (OSF) or internal?

[biu-ethics]: https://research.biu.ac.il/he/node/22446
[biu-edu-form]: https://education.biu.ac.il/files/education/shared/tofes_bakasha_-_etika__2013.pdf
[cfr-46-116]: https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-46/subpart-A/section-46.116
[a13-translation]: https://ebnlaw.co.il/wp-content/uploads/2024/09/Amendment-13-Israeli-Privacy-Law-non-formal-translation.docx
[iapp-a13]: https://iapp.org/news/a/israel-marks-a-new-era-in-privacy-law-amendment-13-ushers-in-sweeping-reform
[pearl-a13]: https://www.pearlcohen.com/the-knesset-enacted-a-key-amendment-to-the-israeli-protection-of-privacy-law-summery-version/
[loc-a13]: https://www.loc.gov/item/global-legal-monitor/2025-11-17/israel-amendment-to-privacy-protection-law-goes-into-effect/
[conti-2011]: https://iris.unitn.it/handle/11572/89854
[gdpr]: https://eur-lex.europa.eu/eli/reg/2016/679/oj
[edpb-3-2018]: https://www.insideprivacy.com/data-privacy/edpb-adopts-final-version-of-guidelines-on-territorial-scope-of-the-gdpr/
[gh-pages-about]: https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages
[gas-web]: https://developers.google.com/apps-script/guides/web
[driveapp]: https://developers.google.com/apps-script/reference/drive/drive-app
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
[drive-zip]: https://workspaceupdates.googleblog.com/2016/08/improvements-to-downloading-files-and-folders-in-Drive.html
[googleone-15gb]: https://support.google.com/googleone/answer/9312312
[drive-limits]: https://support.google.com/drive/answer/37603
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
[phyphox-download]: https://phyphox.org/download/
[bcd-audiosession]: https://github.com/mdn/browser-compat-data/blob/main/api/AudioSession.json
[bcd-screenorientation]: https://github.com/mdn/browser-compat-data/blob/main/api/ScreenOrientation.json
