# stage2: the Stage 2 foundation

`stage2` turns the frozen street-study snapshot into one curated trial table, a canonical fixed-shape
representation of every trial, a cue-free detection baseline and a leak-proof leave-one-person-out
(LOSO) harness. Every later modelling lane builds on it. It reuses `motionlab` (loading, norm_v2,
segmentation ports, gravity propagation) and adds nothing to it.

```
cd analysis
.venv\Scripts\python.exe -m stage2.pipeline all --force   # cache, describe, reference, every lane, numbers.tex
.venv\Scripts\python.exe -m stage2.pipeline --list        # units (core + lanes), their steps, cache status
.venv\Scripts\python.exe -m stage2.pipeline describe      # the descriptive analysis (compute only when stale)
.venv\Scripts\python.exe -m stage2.pipeline figures       # redraw every figure from saved results (no fitting)
.venv\Scripts\python.exe -m pytest tests stage2/tests -q -p no:cacheprovider
```

Section 11 documents the pipeline and the lane conventions (one command per step, compute separate from
presentation, staleness, per-lane files, templates, adding a participant). **The numbers quoted below are for
the 20-person snapshot; the authoritative values are regenerated in `tables/describe_summary.json`,
`results/*_numbers.json` and `results/numbers.tex` (LaTeX macros such as `\DescribeNPersons`), which the report
cites instead.** Paths below are relative to `analysis/reports/stage2/` unless they start with `analysis/` or
`stage2/`.

The scripts also run as files (`python analysis/stage2/scripts/describe.py`). The environment is
frozen in `analysis/requirements-stage2.txt`; lanes must not install packages (aeon is not installable
without downgrading numpy, so MiniRocket has to be written by hand).

| Module | Contents |
|---|---|
| `config.py` | paths (snapshot, curation, cache, outputs; overridable with `STAGE2_*` environment variables), seeds (`SEED = 20261027`, `N_BOOT = 2000`), every threshold below, the code fingerprint |
| `data.py` | curation, trial / person / session tables, flags and gates, the exclusion cascade (`RULES`, N1), the cache (`build_cache`, `load_cache`, `load_tables`, `data_identity`) |
| `standardize.py` | the canonical representation: `stream_from_record`, `canonical` (gravity frame for any bounds), windows, phases, channel table, mirror and screen-flip on arrays |
| `cuefree.py` | cue-free movement detector v0, the evaluation contract (`detect`, `classify`, `evaluate`, `summarize`) and the stress test (`shorten_stillness`, `stress_evaluate`) |
| `cv.py` | LOSO harness (`run_experiment`, `Fold`), metrics, person-cluster bootstrap, `paired_compare`, `k_attempt_curve`, results registry |
| `guard.py` | `LeakageError` and the lock that keeps the cache loaders out of a `fit_predict` |
| `screen.py` | docs/09 D6 wrong-hand screen (`wrong_hand_flags`) and the per-block label-consistency table with the swap signature (`block_consistency`) |
| `qc.py` | numerical helpers of the descriptive analysis (full-window spectra) |
| `plotting.py` | one figure style; `save_fig` writes PDF + PNG and the caption and figure-text registries (process-safe) |
| `locking.py` | the cross-process file lock and atomic file replacement behind every shared registry (section 11) |
| `pipeline.py` | the orchestrator (`python -m stage2.pipeline`), unit discovery, staleness, the `Lane` helper (section 11) |
| `report_export.py` | booktabs tables, numbers files and their LaTeX macros, report-safe tables, collecting report inputs |
| `scripts/` | core units `build_cache.py`, `describe.py` (+ `describe_figures.py`), `reference.py` (+ `reference_figures.py`); `expected_counts.py` (the counts the tests pin) |
| `lanes/` | lane modules `<id>_pipeline.py`, `<id>_figures.py`, `<id>_*.py`, discovered by the pipeline |

Figures are `figures/S2_*.pdf|png` (lanes: `figures/<id>/`), each with a one-line caption and the decision it
justifies in `captions.json`; descriptive tables are `tables/*.csv`; the describe unit's figure inputs, plotted
data and report tables are in `tables/describe/`; every descriptive number is in `tables/describe_summary.json`
and the cited ones in `results/describe_numbers.json`. Persons are P01..P20 only: no name appears in any table,
figure, log or source file (a test scans every output against the names in the raw records, section 12).

## 1. Data and curation

- **Snapshot** `analysis/data/real/2026-10-27` (backend layout), 27 sessions and 878 trial attempts, loaded
  by `motionlab.io.load_dataset`: schema validation, no duplicates, no invalid record.
- **Curation** (binding, from the researcher): `analysis/data/curation/sessions.json` maps every
  session to a person P01..P20 and includes 24 sessions of 20 persons. Only its `sessions` list is read
  (`data.load_curation`); its free-text reasons start with first names and are replaced by fixed codes;
  the private name file next to it is never opened (`config.assert_not_private`).
- **Grouping is by `person_id`, never by the app's `participant_code`.** One code holds two persons (P01
  and P13 used the same phone) and one person holds two codes (P14). `motionlab` keys participant
  profiles by code, so for that shared code its trial table carried the newest profile (P13's) on P01's
  trials too. `stage2` therefore takes demographics from each session's own snapshot, keeps them in the
  person table only (never in the trial table, so they cannot slip into a model) and checks that every
  person's sessions agree (`profile_consistent`: all 20 do).
- **P20** (added last): one complete Android street session, 32 attempts of which 29 accepted, identified by the
  researcher (the app session carried no name). With P20 removed, every count of the 19-person snapshot is
  reproduced exactly (`python -m stage2.scripts.expected_counts --without P20`).
- Persons with several sessions: P01 (the protocol 1.0.0 pilot, app 0.1.0, plus a street visit), P09
  (2), P14 (3). Partial sessions are kept: the plan balances hands within every tier, so even an early
  stop contributes both hands (P04 stopped after tier 1 with 4 left and 4 right trials).
- **Exploratory persons** (PD01, ...) are not part of any of this: they live outside the snapshot
  (`analysis/data/exploratory/`, `curation/exploratory.json`), only lane x1 reads them, they are never trained
  on, tuned with, validated on or scored by a model, and `tests/test_exploratory_guard.py` checks that none of
  their ids reaches the main snapshot, curation or cache.

**People** (table `participants.csv`, figure `S2_01_demographics`, LaTeX `tables/describe/describe_participants.tex`):
20 persons, age 25.1 ± 2.9 years (19–29), 12 women and 8 men, 18 right-handed and 2 left-handed (P11, P17),
usual phone hand right 18 / both 2, height 168.8 ± 8.8 cm (n = 17, 152–187). 12 persons on Android (15
sessions, 14 device models in all) and 8 on iPhone (9 sessions).

**Completion** (`completion.csv`, `visit_durations.csv`, `S2_03_completion_durations`): 23 street visits,
19 completed and 4 ended early; the optional bonus tier was taken in 7 of the 19 completed visits; 1.26
attempts per accepted trial. A completed visit took 9.7 min (median; IQR 7.8–11.0), not the planned 5:
tier 1 2.4 min, tier 2 3.1, tier 3 1.8, bonus 2.0.

## 2. Exclusion cascade

Rules apply in order, and an attempt is charged to the first rule it fails (`data.RULES`,
`tables/exclusion_flow.csv`, `S2_02_exclusion_flow`, `describe_exclusion_flow.tex`). E3 diverts rather than drops.

| Step | Rule | Removed | Remaining (persons) | Justification |
|---|---|---|---|---|
| start | every attempt in the snapshot | | 878 (21) | 27 sessions |
| E0 | session excluded by the researcher | 42 | 836 (20) | 3 sessions: an earlier incomplete run superseded by the person's later run; two runs of one person that did not work |
| E0s | simulated record (S1) | 0 | 836 | checked, none |
| E1 | attempt not accepted (docs/02 T1–T3) | 172 | 664 | 79 no movement within the limit, 79 still moving at the limit, 4 revoked ("redo last movement"), 3 not confirmed at tap review (pilot), 3 auto-rejected for a wrong grip, 2 aborted, 1 flagged by the participant, 1 interrupted (`exclusion_E1_breakdown.csv`) |
| E2 | screen-away grip (protocol 1.0.0 only) | 0 | 664 | the only `*_away` attempt was already out (flagged at E1); street visits never collect them, the analysis synthesises them (`screen_flip_channels`) |
| E3 | tier-0 free-choice trial | 23 → natural-use set | 641 | its hand is reported after the movement, not instructed: never trained on, scored separately |
| E4a | no approach window | 0 | 641 | every complete attempt has a stored segment |
| E4b | approach shorter than 0.25 s (gate G2) | 2 | 639 | a reach from arm's length cannot be that short: a fragment was segmented (`S2_08_durations_peaks`) |
| E4c | phone did not turn: peak \|ω\| < 30 deg/s | 11 | 628 | every reach rotates the forearm; 30 deg/s is the app's onset floor and gate G2 of docs/09; the natural-use set loses 1 here |
| E4d | sampling gap > 100 ms within [onset − 0.5 s, end + 0.5 s] | 30 | **598 (20)** | gaps up to 100 ms (≤ 5 samples) are interpolated; longer ones near the movement remove part of it (`S2_05_sampling_gaps`). 26 of the 30 are iOS (see section 8) |
| E4e | page hidden, rate < 50 Hz, no gyroscope, unsupported gravity sign (T4, T5, T7, S3) | 0 | 598 | checked, none |
| N1 | natural use: one free-choice trial per person (the earliest that passed every rule) | 2 → `natural_use_repeat` | 598 | the free trials of repeat visits (P09's second, P14's third session) are reported apart (section 5.3) |

**Primary set: 598 trials of 20 persons in 24 sessions** (302 left, 296 right; 383 Android, 215 iOS;
8 to 45 trials per person, median 28, both hands for everyone). **Natural-use set: 20 trials, one per person,
all 20 with the right hand** (too unbalanced for anything but an accuracy), plus 2 repeat-visit free trials in
`natural_use_repeat` (1 left, 1 right) reported apart.

The cache also holds the arrays of every excluded complete attempt, so sensitivity analyses (for example with
the E4d trials) need no rebuild: `load_cache(sets=None)` and filter on `excl_rule`.

### 2.1 Flags and gates: scored apart, never exclusions

Two kinds of trial-level columns are kept in the trial table and never remove a trial
(`tables/flags_and_gates.csv`, `S2_18_flags_gates`, `describe_flags_gates.tex`; test
`test_flags_are_never_exclusions_and_gates_are_scored_apart`):

- **Flags** (docs/02 T8–T12, `data.FLAG_COLUMNS`; reported, used for stratification only):

  | Column | Meaning | Primary trials (persons) |
  |---|---|---|
  | `grip_mismatch` (T8) | the grip before the onset is not the instructed one, by the app or by the standardised up vector of the 300 ms before the onset; the hand label is kept | 23 (3) |
  | `seg_disagree_100ms` (T9) | offline_v1 and the stored online_v1 onsets differ by more than 100 ms | 115 (19) |
  | `clip_candidate` (T10) | \|accg\| > 3.5 g or \|ω\| > 950 deg/s somewhere in the raw recording | 8 (2) |
  | `repeated_samples` (T11) | isolated duplicated snapshots were dropped before resampling | 126 (20) |
  | `false_start` (T12) | the hold restarted before GO | 128 (20) |
  | `screen_rotated` | the app warned of a screen rotation during the movement | 2 (1) |

- **Predictor gates** (docs/09 D5, `data.GATE_COLUMNS`; True = the predictor would score the attempt):
  `gate_g1` (rate ≥ 50 Hz, no gap > 50 ms in [onset − 0.3 s, end + 0.3 s], page visible, gyroscope present;
  16 primary trials fail), `gate_g2` (0.25 s ≤ duration ≤ 8 s and peak \|ω\| ≥ 30 deg/s; 13 fail, 12 of them
  instructed slow reaches), `gate_pass` (both; 28 fail). Failing trials stay in training and in the headline
  metrics; whenever `gate_pass` is a column of `meta`, `run_experiment` also reports the metrics on the trials
  that pass both gates (`gated_metrics`, the registry's `gated_accuracy`), as the predictor would see them.
  Why the gates are not exclusions here is in section 5.4.

## 3. The canonical representation: decisions and evidence

| Decision | Choice | Evidence |
|---|---|---|
| Frame | W3C device frame (x right edge, y top edge, z out of the screen) after `motionlab` norm_v2: per-session accelerometer sign and rotationRate mapping | Every included session's sign is supported: 23 by the engine prior confirmed by the trial votes, the pilot by its guided calibration; Android +1, iOS −1 as expected (`normalisation_per_session.csv`). W3C mapping in all 24: from the pilot's calibration tilt check, from the gyro-gravity closure in 21 sessions (median 1.0° against 13.4° for the runner-up), and assumed in two short sessions. On real data the mirror predicted by the conventions, σ* = (−,+,+;+,−,−), maps right-hand onto left-hand waveforms better than any of the 64 sign patterns, on both platforms (`sign_patterns.csv`, `S2_12_sign_patterns`) |
| Heading-free frame | gravity frame from gravity_v1: the rest gravity before the onset and after the end, propagated with the gyroscope and blended; up vector u = g/\|g\|; vertical and horizontal parts of a and ω, yaw rate ω·u, its integral, tilt from the start | gyro-gravity closure median 1.1°, 7 % above 3° (`S2_17_gravity_closure`); no rest-window fallback was needed |
| Gyro bias | none subtracted (a deviation from docs/09 D9, section 5.4) | norm_v2; street sessions have no rest calibration and production has none either; the closure above shows it is not needed |
| Units | m/s² and deg/s (time in ms on the trial clock) | the record units; deg/s keeps ω readable |
| Resampling | linear onto a uniform 60 Hz grid after dropping isolated duplicated samples | both platforms deliver 60 Hz: median interval 16.7 ms (Android) and 17.0 ms (iOS), 99th percentile 18.5 / 18.0 ms (`sampling_quality.csv`, `S2_05_sampling_gaps`); upsampling would add no information and production sees the same rate |
| Low-pass | zero-phase Butterworth, order 4, 15 Hz, run by run over finite samples | 99.2 % of the gyroscope and 98.4 % of the accelerometer power of the approach lie below 15 Hz (raw data, 598 trials, every window tapered and transformed over its full length; `S2_07_spectrum`). Zero phase is legitimate in production: the predictor processes a recorded buffer |
| Precision | native on both platforms (Android values sit on Chrome's 0.1 grid, iOS values are unrounded); `quantize_ios=True` is ablation A3 | the rounding noise (SD 0.1/√12 = 0.029) is far below the hand-held noise in the still window before GO: specific force SD 0.154 (Android) / 0.147 m/s² (iOS), angular velocity 2.5 / 2.8 deg/s (`S2_06_noise_vs_quantisation`) |
| Gaps | ≤ 100 ms interpolated; longer gaps stay NaN in the stream (`valid` False) and exclude a trial when they fall near the movement (E4d) | Android gaps (> 50 ms) last 51–152 ms (median 64 ms; 34 gaps in 33 attempts); iOS gaps are ~0.3 s (median 314 ms), 89 of them in 73 iOS attempts (`gaps.csv`) |
| Oracle approach window | the app's stored online_v1 segment (`onset_ms`, `end_ms`): onset back-dated to the end of the stillness, end back-dated to the start of the final stillness at the nose (the end cue follows 250 ms later) | exists for every complete attempt and is exactly reproducible (`motionlab.segment.rerun_online_v1`). offline_v1 finds no end in 19 % of street attempts (0 % of the pilot, whose 1-s post-roll had no return of the arm); where both exist they agree within −56..−10 ms at the onset and −130..+90 ms at the end (IQR; `segmentation_agreement.csv`, `S2_09_segmentation_contact`). offline_v1 bounds are kept (`off_onset_ms`, `off_end_ms`) for ablation A5 |
| Contact | not used as a bound | 72 % of screen-toward attempts record a touch; the first touch falls at 73 % of the oracle window (IQR 45–85 %), 333 ms before its end (medians): the last quarter is the phone settling against the nose |
| Fixed window | −0.5 s .. +2.5 s around the oracle onset, 180 samples at 60 Hz; samples from the end cue on are frozen (carry-forward) and masked | 73 % of approaches last ≤ 2.5 s (median 1.65 s); after the end cue the participant returns the arm, which must never enter (section 6) |
| Time normalisation | 64 samples from onset to end, inclusive | about 26 ms per sample at the median duration; `yaw_cum` and `tilt_from_start` are referenced to the onset |
| Return window | [return onset, return end] as recorded (schema 1.2), time-normalised to 64 samples, its own gravity reference; labelled separately | available (observed, gap-free) for 288 of the 598 primary trials (`ret_ok`) |

The same `standardize.canonical(stream, onset_ms, end_ms)` runs on oracle bounds (training) and on
bounds found by a cue-free detector (production); only the bounds differ. Test
`test_mirror_is_exact_on_the_cached_channels` shows that standardising a mirrored record equals mirroring
the cached arrays (`mirror_channels`), so mirror augmentation can act on the cache directly.

### Channels (`standardize.CHANNELS`; arrays are (N, C, T) float32)

| Channel | Frame | Unit | Mirror | Screen flip | Meaning |
|---|---|---|---|---|---|
| `acc_x`, `acc_y`, `acc_z` | device | m/s² | −, +, + | −, +, − | linear acceleration a = f − g |
| `gyro_x`, `gyro_y`, `gyro_z` | device | deg/s | +, −, − | −, +, − | angular velocity |
| `up_x`, `up_y`, `up_z` | device | 1 | −, +, + | −, +, − | unit up vector from gravity_v1 |
| `gyro_mag`, `acc_mag` | gravity | deg/s, m/s² | + | + | magnitudes |
| `acc_vert`, `acc_horiz` | gravity | m/s² | + | + | a·u and \|a − (a·u)u\| |
| `yaw_rate` | gravity | deg/s | **−** | + | ω·u: rotation about gravity, + = counter-clockwise seen from above (chiral) |
| `gyro_horiz` | gravity | deg/s | + | + | tilting rate \|ω − (ω·u)u\| |
| `yaw_cum` | gravity | deg | **−** | + | ∫ ω·u dt from the window onset (chiral) |
| `tilt_from_start` | gravity | deg | + | + | angle between u(t) and u at the window onset |

"Mirror" is the sign under the left/right mirror (a → S a, ω → −S ω, S = diag(−1, 1, 1)): the label flips
too. "Screen flip" is the phone turned 180° about its y axis (F = diag(−1, 1, −1) on every vector; the
label stays). Device channels depend on the grip; gravity channels do not.

### The cache (`analysis/data/stage2_cache/`, git-ignored)

| File | Content |
|---|---|
| `trials.pkl` / `.csv` | one row per attempt (878): ids, `person_id`, `participant_code`, `session_id`, platform, engine, device, app / protocol / schema versions, condition (hand, `hand_source`, orientation, speed, start position, variant, posture, tier, block), outcome, timing (GO, onset, end, end cue, first touch, return onset / end, stop), quality, `motionlab` segmentation and peaks, offline_v1 bounds, the QC of `standardize_trial` (`duration_ms`, `peak_gyro_dps`, `closure_deg`, gap counts, `first_half_yaw_deg`, `net_yaw_deg`, return status), flags and gates (section 2.1), `analysis_set` (`primary` / `natural_use` / `natural_use_repeat` / `excluded`), `excl_rule`, `excl_detail`, `row` (index into the arrays) |
| `persons.pkl` | one row per person: profile, devices, sessions, counts (attempts, accepted, primary, natural use, primary left / right), tiers reached |
| `sessions.pkl` | one row per session (no user agent), curation code, normalisation |
| `exclusion_flow.pkl`, `exclusion_e1.pkl` | the cascade and the E1 reasons |
| `arrays.npz` | for the complete attempts of the included sessions: `approach` (N, 17, 64), `fixed` (N, 17, 180), `fixed_mask` (N, 180), `fixed_phase` (N, 180), `ret` (N, 17, 64), `ret_ok` (N) |
| `streams.npz` | the full standardised recordings (60 Hz, W3C device frame, not windowed): `t`, `f` (specific force), `acc` (browser linear acceleration), `gyro`, `valid`, `phase`, ragged with `offsets` |
| `meta.json` | cache version, data identity, snapshot fingerprint, git commit, code fingerprint, every parameter, channel table, flag and gate definitions, counts (section 12) |

Phases (`standardize.PHASES`): 0 pre, 1 approach, 2 hold (end to end cue), 3 post-cue, 4 return, 5 after
the return.

## 4. Cue-free detection, baseline v0 (`cuefree.py`)

In the inference app nothing marks the start. v0 takes the first sustained movement (smoothed |ω| > 30
deg/s for 100 ms) that follows at least 1 s of stillness (|ω| < 20 deg/s, gyroscope only) and ends it at
the next 250 ms of stillness. Thresholds are the app's own (still_gyro_dps, onset floor, end_sustain).
Gyroscope only, because requiring |a| < 0.5 m/s² too leaves 90 % of the approaches with at least 1 s of
stillness before them (median 1.8 s), against 94 % with the gyroscope alone (median 2.4 s). Stillness is
measured in v0's own activity signal: the still run whose last still sample lies within 300 ms of the oracle
onset (`describe_summary.json` `stillness_before_onset`).

On the 598 primary recordings (`cuefree_v0_summary.csv`, `S2_16_cuefree_v0`, `describe_cuefree_v0.tex`) v0 finds:

| What v0 finds first | Share |
|---|---|
| the approach (IoU ≥ 0.5 with the oracle window) | 53 % |
| part of the approach (IoU < 0.5) | 12 % |
| an earlier movement (the arm settling after the previous trial, a fidget) | 28 % |
| the return of the arm | 2 % |
| nothing | 5 % |

The approach is among all of its detections in 72 % of recordings. When v0 finds the approach, its
onset is 48 ms late (median; 90th percentile of the absolute error 183 ms). The main failure is
therefore an earlier movement, not the return. `cuefree.evaluate(data)` and `cuefree.summarize` are the
contract a better detector must report against.

**Caveat: the stillness before every recorded approach was enforced by the protocol, so these shares are
upper bounds.** Before each GO the collection app waited for 400 ms of stillness, then a 400–900 ms
foreperiod, then the reaction time (median 575 ms from GO to the onset). Production has no GO and no
hold-still gate, so a user may start moving without holding still first. The **stress evaluation**
(`cuefree.stress_evaluate`, `cuefree_v0_stress.csv`, `S2_19_cuefree_stress`, `describe_cuefree_stress.tex`)
cuts the still run before each approach to `keep_s` seconds (the samples right after the preceding activity
are removed and the rest of the recording moves back, so the earlier movement now ends `keep_s` before the
approach) and runs v0 again:

| Stillness kept before the approach | recorded | ≥ 1 s | 0.75 s | 0.5 s | 0.25 s | 0 s | random U(0, 1.5 s), seeded |
|---|---|---|---|---|---|---|---|
| v0 finds the approach first | 53 % | 53 % | 3 % | 2 % | 2 % | 10 % | 24 % |

Below its own 1-s rule v0 can no longer separate the approach from what came before. A production detector
must not rely on a long stillness, or the inference app must ask for one and check it. The cue-free lane reports
both the recorded shares and the stress curve.

## 5. Evaluation harness (`cv.py`)

`run_experiment(name, fit_predict, X, meta, ...)` is the only way a model is evaluated: leave-one-person-out,
one outer fold per person (20 folds). Lanes call it through `Lane.run_experiment` (section 11).
`fit_predict(fold)` returns P(right) for the test rows, or `(P(right), info)` (info, for example the chosen
hyperparameters, is stored per trial).

### 5.1 Exactly what the harness enforces

Each item raises (`LeakageError` or `ValueError`) unless stated otherwise, and is unit-tested in
`tests/test_cv.py`:

1. **Rows are curated trials of persons.** `meta` needs `trial_id`; the ids are matched against the cache's
   curated trial table (`data.canonical_trial_map`). A row whose `meta.person_id` differs from the curated
   person raises, so grouping by the app's participant code or by session is impossible; duplicated ids and a
   mix of cached and unknown ids are refused.
2. **Only the primary set is a row** unless the caller opts in (`sets=`; for example `'excluded'` for a
   sensitivity analysis). `natural_use` and `natural_use_repeat` trials can never be rows.
3. **Folds are persons.** One outer fold per person (sorted ids). `groups=` may merge persons
   (leave-group-out, with a warning) but never split one. Every fold is asserted disjoint in groups, persons and
   rows, and its test person is checked to be absent from its training rows.
4. **The training side** gets copies of the training rows only (`X_train`, `y_train`, `groups_train`,
   `meta_train`), after the docs/09 D2 cap: at most `cap_per_person` (40) trials of one person, a seeded,
   hand-balanced choice that is the same in every fold; the held-out person is never capped. The persons and
   rows dropped are recorded (`info.capped_training_rows_per_fold`).
5. **The test side** gets `X_test` and `meta_test` with only `TEST_META_COLUMNS` (`row`, `person_id`,
   `platform`, `engine`, `device_model`, `app_version`, and the sensor-derived `duration_ms`, `peak_gyro_dps`,
   `peak_acc_ms2`, `closure_deg`, `approach_valid_frac`, `motion_hz`): no label, no instructed condition, no
   block or trial order, no lookup key (no `trial_id`), no demographic. Asking for any of them raises. The test
   rows come in a seeded random order with an opaque `row` index and are mapped back.
6. **X carries no label.** A DataFrame column or dict key named like a label, condition, lookup key or
   demographic, or containing a label-like token (`hand`, `label`, `target`, `handedness`), raises; so does any
   two-valued column (of an array: any flattened column) that equals y or 1 − y.
7. **The cache is locked while `fit_predict` runs** (`guard`): `load_cache`, `load_tables` and
   `canonical_trial_map` raise inside it.
8. **Test-set dependence is detected in every fold** (`leak_check='all'`, the default): each fold is re-run on a
   random half of its test rows in a new order, and predictions for the shared trials must not move
   (atol 1e-9). A scaler fitted on `X_test`, per-person test-time normalisation or a row-position feature raises;
   a non-deterministic `fit_predict` gets a warning and the record says `nondeterministic` (seed it with
   `fold.seed`). The checked folds and each outcome are stored (`info.leak_check_per_fold`). `'sample'` checks 3
   seeded folds, `'off'` none (both recorded).
9. **Predictions are validated:** one finite P(right) in [0, 1] per test row.
10. **Nested tuning has its own splitter:** `fold.inner_splits()` is GroupKFold (5, or fewer when there are
    fewer training persons) over the training persons, checked for person overlap.
11. **Configurations compared on the outer folds are declared:** `n_configs_tried` (≥ 1, default 1 =
    pre-specified) is stored in the result and the registry (section 5.4).
12. **Natural-use rows are scored, never trained on** (`extra=(X_nat, meta_nat)`): they are matched to the cache
    like the rows (only `natural_use` / `natural_use_repeat` allowed), join the test rows of their person's fold,
    are scored by that fold's model and are reported apart (`extra_predictions`, `extra_metrics`); a trial cannot
    be both a row and an extra row.
13. **Metrics and records:** accuracy (primary), balanced accuracy, ROC AUC, log loss, Brier, ECE (10 equal-mass
    bins), reliability data and per-person accuracy, each with a 95 % CI from 2000 person-cluster bootstrap
    resamples; the k-attempt curve on the same persons for every k (those with ≥ 5 trials of each hand: 19 of
    20 now); the gated metrics (section 2.1); `results/<name>.json` (per-trial predictions with `trial_key`, all
    of the above, provenance: section 12) and one registry row. The registry update holds a cross-process lock
    and replaces the file atomically (`locking.py`), so parallel lanes or workers never lose a row.
14. **Ties count half** (since 2026-10-28, `cv.credit`): P(right) > 0.5 answers right, < 0.5 left, and a tie
    (0.5 up to rounding, |P - 0.5| <= `cv.TIE_ATOL` = 1e-12: an abstention or a constant predictor) earns half
    credit in every accuracy: overall, balanced, gated, natural use, per person, the k-attempt curve (a mean of
    0.5) and `paired_compare`. When a model abstains, `metrics` adds `coverage` (share answered) and
    `accuracy_answered`, and the registry row `coverage`, `accuracy_answered`, `extra_coverage`,
    `extra_accuracy_answered`. Before, a tie counted as a right-hand answer, so on the all-right natural-use set
    every abstention scored as correct (a constant 0.5 scored 100 %). `python -m stage2.scripts.rescore --write`
    re-scored the saved results from their saved predictions (no refit) and lists every changed result with its
    old and new numbers in `reports/stage2/maintenance/rescore_ties.csv|json`; each changed result records
    `info.rescored` and its rule (`info.scoring`), and so does every result run from now on. A saved result
    without `info.scoring` holds no tie, so both rules give it the same numbers (the re-score left it
    byte-identical).
15. **ECE caveats.** The percentile person-bootstrap CI of the ECE (`ci['ece']`) is biased upwards (resampling
    persons adds binning noise to every bin) and can exclude its own estimate: never cite it; judge an ECE against
    its distribution under perfect calibration or compare ECEs through paired differences. The pooled equal-mass
    ECE (and AUC) of a predictor whose P is constant within a fold (a rule scoring its training hit rate) mostly
    sorts the folds and is not comparable with a fitted model's.

### 5.2 What it cannot see (rules the lanes keep)

- **Choosing among configurations on the outer folds.** The best of K outer-fold scores is optimistic. Headline
  numbers come from nested tuning (`fold.inner_splits()`) or from pre-specified configurations; a comparison of K
  configurations on the outer folds passes `n_configs_tried=K` and is reported as exploratory.
- **Preprocessing fitted outside `fit_predict`.** Anything that spans trials (scalers, PCA, feature selection,
  MiniRocket biases, vocabularies, thresholds) must be fitted inside it on the training rows; a scaler fitted on
  all rows before `run_experiment` leaks the held-out person's feature statistics and is not detected. Per-trial
  transforms (a function of one trial alone) may be computed beforehand.
- **Closures over outside data.** A `fit_predict` that captured labels, demographics or instructed conditions
  before the call and looks them up (for example by `person_id`) is not detected. Model inputs are
  sensor-derived only, plus platform or device; demographics and instructed conditions only stratify results.
- **A dependence on the test set that survives the random half** of check 8 is not caught (it would be rare).

### 5.3 The natural-use headline rule

The natural-use headline is **one first free-choice trial per person**: rule N1 keeps, per person, the earliest
free-choice trial that passed every rule (ordered by session start, the person's session order, trial index and
attempt), because it is that person's first, unpractised movement with a hand they chose, the closest thing in the
data to the PI's first attempt (docs/09 D4). Later free trials (repeat visits, practised) form
`natural_use_repeat` and are reported apart, never pooled into the headline. Natural-use trials are never trained
on: `run_experiment` refuses them as rows and scores them through `extra=`. Today the headline set has 20 trials,
all with the right hand, so it supports an accuracy with a confidence interval (one trial per person) and nothing
that needs both classes (AUC, balanced accuracy, calibration). Its accuracy is right-hand recall only: a constant
'right' answer scores 100 %, so it is never a balanced accuracy; an abstention (P(right) = 0.5) earns half credit
(section 5.1 item 14) and the result then also reports `coverage` and `accuracy_answered`. Tests:
`test_natural_use_set_is_the_reported_choice_trials`, `test_natural_use_headline_is_one_first_trial_per_person`.

### 5.4 Deviations from docs/09 and why

| docs/09 | The plan | Stage 2 | Why |
|---|---|---|---|
| D2 person key | `person_key` = the earliest P-code, merged through an alias table | the researcher's curated `person_id` (P01..P20) | one app code holds two persons and one person two codes (section 1); the curation is the ground truth, and `run_experiment` refuses any grouping that splits a person |
| D2 cap | at most 40 trials of one person per training fold, seeded | the same cap (`config.CAP_PER_PERSON`), seeded per person, hand-balanced, the same rows in every fold, never applied to the held-out person | the cap stops one person dominating training; balance keeps the capped person's hands even. It affects only persons with more than 40 primary trials (now one: P01, 45), recorded per result |
| D3 locked test | 25 % of persons by hash, opened once for one frozen model | **waived** (`info.locked_test`) | the researcher's instruction: evaluate every model by leave-one-person-out on all persons (a 25 % test set would hold 5 persons). Without an untouched test set, model choice must not use the outer folds: nested tuning or pre-specification for headline numbers, `n_configs_tried` for any outer-fold comparison (section 5.2) |
| D5 exclusions vs gates | no new exclusion rules; movement checks are predictor gates G1, G2, applied in evaluation exactly as the predictor will (G1, G2 failures not scored) | E4b (duration < 0.25 s), E4c (peak \|ω\| < 30 deg/s) and E4d (gap > 100 ms within 0.5 s of the movement) **are exclusions**; G1 and G2 are evaluated as columns but **do not exclude**: their failures stay in training and in the headline, and every result also reports the gated metrics | E4b and E4c are G2's own lower bounds: such a segment is a fragment, not a reach (`S2_08`), and has no valid representation; E4d's holes remove part of the movement and cannot be interpolated (`S2_05`). The remaining gate failures are valid movements (G1: gaps of 51–99 ms, interpolated; G2: 12 of 13 are instructed slow reaches over 8 s), so keeping them makes the headline the harder, conservative number, and `gated_metrics` gives the predictor's view. G2's offline_v1 clause is not applied (offline_v1 finds no end in 19 % of street attempts; D10) |
| D6 wrong-hand screen | inside each outer training fold | implemented (`screen.wrong_hand_flags`) and run as a sensitivity analysis (`sens_d6_screen`); not part of the reference | it changed the reference by −0.3 points (paired sign-flip p = 0.59; `results/reference_numbers.json`); the labels are never changed |
| D9 gyro bias | subtract the pre-GO mean rate when below 3 deg/s | none subtracted | street sessions have no rest calibration, production has no GO, and the gyro-gravity closure is 1.1° (median) without it (`S2_17`) |
| D10 deployed segmentation | offline_v1 | the oracle is the stored online_v1 segment; offline_v1 bounds are kept for ablation A5 | offline_v1 finds no end in 19 % of street attempts (`S2_09`) |
| D15, D16 model selection and schemes | development CV (stratified group k-fold 5 × 5, LOSO, ...), then the locked test | LOSO only; nested tuning inside each fold (`fold.inner_splits()`), `n_configs_tried` for outer-fold comparisons | the researcher's LOSO-only instruction; no locked test (D3) |
| D4 natural use | choice trials scored apart | plus N1: one first free trial per person in the headline, repeats apart | section 5.3 |

### 5.5 Reference

Reference row in the registry: `example_yaw_logreg`, the example of section 9 (core unit `reference`,
`python -m stage2.pipeline reference`). It is a logistic regression on the cumulative yaw at 8 instants,
mirror-augmented, with LOSO accuracy 77.9 % [73.1, 82.9] over 20 folds, AUC 0.855, 75 % on the 20
natural-use trials, 87.2 % with k = 3 and 90.9 % with k = 5 attempts (`results/reference_numbers.json`). It is
a sanity floor, not a model choice. Its label-noise variants (without the swap-signature blocks in training, or
in training and test, or with the D6 screen) stay within ±0.3 points of it, none beyond its paired 95 % CI
(`S2_21_reference_sensitivity`).

## 6. What the descriptive analysis shows

**The hand is in the chiral, time-resolved gravity-frame signal** (`S2_11_class_means`,
`S2_13_first_half_yaw`, `first_half_yaw_per_person.csv`, `describe_first_half_yaw.tex`).
- Early in the reach, right-hand approaches turn counter-clockwise about gravity (seen from above) and
  left-hand approaches clockwise. The mean yaw rate peaks near 25 % of the approach.
- The pilot's first-half yaw finding holds on all 20 persons:
  - pooled AUC 0.85 (Android 0.86, iOS 0.84);
  - within-person AUC median 0.90 (minimum 0.63), above 0.5 for all 20 persons and at least 0.8 for 15;
  - its sign alone is right in 79 % of trials;
  - medians are +5.3° for the right hand and −3.8° for the left.
- Unlike in the pilot, net yaw also separates the hands (pooled AUC 0.77). Second-half yaw does not
  (0.49).

**Mirror symmetry holds for the chiral channels** (`mirror_symmetry.csv`, `describe_mirror_symmetry.tex`).
- The mirrored left-hand mean yaw rate lies close to the right-hand mean: RMS difference 3.4 deg/s,
  against 12.2 deg/s between the unmirrored means. For 18 of the 20 persons the right-hand mean is
  closer to the mirrored left-hand mean.
- The even channels (tilting rate, vertical and horizontal acceleration, tilt) barely differ between
  hands.
- The left-hand yaw is smaller early in the reach, which is a dominance-type asymmetry: 18 of the 20 persons are right-handed.

**iOS and Android agree after standardisation** (`S2_14_platforms`, `platform_differences.csv`, `describe_platforms.tex`).
- The yaw signs per hand are the same on both platforms, which confirms the sign normalisation.
- iOS approaches are slightly shorter (median 1.42 s against 1.80 s), turn slightly faster (peak |ω| 100
  against 94 deg/s) and close the gravity estimate better (0.8° against 1.3°).

**Durations** (`durations_by_speed_platform.csv`, `describe_durations.tex`). The instructed speed separates durations
only loosely: fast 1.42 s, natural 1.63 s, slow 5.52 s (medians). The median reaction time from GO to onset is 575 ms.

**The return looks like the other hand** (`S2_15_return_vs_approach`, `return_risk.csv`, `describe_return_risk.tex`).
On the 288 trials with a clean return, the approach rule "right iff first-half yaw > 0" is right on 76 % of the
approaches and on only 17 % of the returns (return first-half-yaw AUC 0.11). The return is a reversed
approach. A detector that hands a return to an approach model therefore gets a confidently wrong
answer, worse than chance.

**Labels are consistent, with one exception** (`label_consistency_blocks.csv`, `S2_20_label_consistency`). One
pair of adjacent blocks (P08's blocks 1–2) shows the swap signature: every trial of both blocks contradicts its
label. The blocks are found from the data (`screen.block_consistency`, `reference.swap_blocks`), the labels are
never changed, and the sensitivity analyses of section 5.5 show they do not move the reference.

## 7. Observations that should shape modelling

1. **Use time-resolved chiral gravity-frame inputs** (`yaw_rate` and `yaw_cum` over time). Whole-movement
   integrals lose part of the evidence. The device channels carry the grip and help only with grip
   augmentation or leave-one-grip-out checks.
2. **Antisymmetric models and mirror augmentation are justified.** The chiral channels are close to mirror
   images and the even channels hardly differ. A mirror-symmetric model also cannot learn the dominance
   asymmetry, which with 2 left-handers would not transfer.
3. **Production needs a detector that rejects returns and earlier movements, without relying on a long
   stillness** (section 4). The return predicts the wrong hand 83 % of the time. Candidate rules:
   - prefer the movement that starts from the still "arm out" pose;
   - require an end pose that is held;
   - score candidates with a separate approach-vs-return classifier.

   The return also carries hand evidence in reverse, so a separate return head (ablation A10) could add
   evidence when both are seen.
4. **Accuracy rises with attempts.** The example floor goes from 77.6 % (k = 1) to 90.9 % (k = 5), so the
   inference app should ask for several movements and average them.
5. **Platform is a reporting stratum, not a feature.** E4d removes 26 iOS trials and 4 Android trials, so
   report per platform and add a leave-one-platform-out check.
6. **Mind the small numbers.** Per-person CIs are wide: 8 to 45 trials per person, and P04 has only 8. The
   natural-use set is 20 right / 0 left, so treat it as a single accuracy with a CI.

## 8. Caveats

- **iOS sampling stalls.** 66 iOS attempts contain a 0.3-s sampling gap (median 318 ms). These gaps start
  a median 4.2 s after the recording starts (IQR 3.8–4.6 s), and only 25 % follow an audio cue within
  0.4 s. This points to a recurring main-thread stall in the collection app, not to the sensor. Where
  such a gap meets the movement it removes the onset, so E4d excludes 26 iOS trials (11 % of the iOS
  candidates). The predictor must keep the main thread free while it records.
- **The cue-free shares are upper bounds** (section 4): the recorded stillness before each approach was
  enforced by the protocol.
- **offline_v1 needs a longer post-roll than the street protocol gives.** It finds no end in 19 % of
  street attempts, so docs/09 D10, which makes it the deployed segmentation, needs revisiting by the
  cue-free lane.
- **Visits run long.** They take about 10 minutes, not 5, and 4 of 23 street visits ended early.
- **Report colour.** The hand colours are left `#D1495B` and right `#00739A` (`report/preamble.tex` defines
  `handright` as `00739A`; the old `#00798C` failed the chroma floor and sat in the colour-blind warning band).

## 9. API for later lanes

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from stage2.data import load_cache
from stage2.cv import run_experiment
from stage2.standardize import mirror_channels

d = load_cache()                                   # primary set: 598 trials of 20 persons
nat = load_cache(('natural_use',))                 # 20 free-choice trials, one per person (self-reported hand)
k = d.channels.index('yaw_cum')
feat = lambda X: X[:, k, ::8]                      # cumulative yaw about gravity at 8 instants

def fit_predict(fold):                             # only the training persons' labels are visible
    Xtr = np.concatenate([fold.X_train, mirror_channels(fold.X_train)])     # mirror augmentation
    ytr = np.concatenate([fold.y_train, 1 - fold.y_train])
    m = LogisticRegression(max_iter=1000).fit(feat(Xtr), ytr)
    return m.predict_proba(feat(fold.X_test))[:, 1]

res = run_experiment('example_yaw_logreg', fit_predict, d.approach, d.trials, extra=(nat.approach, nat.trials))
print(res.metrics['accuracy'], res.ci['accuracy'], res.extra_metrics['accuracy'], res.leak_check)
```

**`Stage2Data`** (`load_cache(sets=('primary',))`; `sets=None` returns every cached attempt):
- `trials`: the trial table rows, aligned with the arrays;
- `approach`, `fixed`, `fixed_mask`, `fixed_phase`, `ret`, `ret_ok`: the arrays;
- `channels`: the channel names; `channel_rows(names)` gives their indices;
- `y`: 1 = right hand;
- `groups`: `person_id`;
- `stream(i)`: the full standardised recording of row i;
- `subset(mask)`: a filtered copy.

**`run_experiment` and `fit_predict`.** `X` can be an array, a DataFrame, or a dict of arrays aligned
with `meta`. `fit_predict` may return `(p, info)`; `info` (for example the chosen hyperparameters) is
stored per trial. Lanes call it as `Lane('<id>').run_experiment(...)` (their own registry, section 11).

**Other entry points:**
- `cv.paired_compare(res_a, res_b)` compares two experiments (per-person differences, paired person bootstrap,
  exact sign-flip test); vectorised (about 0.2 s), output-identical to the former per-resample loop, so lanes need
  no copy of it;
- `cv.credit(y, p)` is the per-trial accuracy credit (a tie earns half) behind every accuracy; `cv.is_tie(p)`;
- `cv.load_result(name)` reloads a saved result;
- `cuefree.evaluate(d)` scores a detector against the oracle window; `cuefree.stress_evaluate(d)` shortens the
  stillness first;
- `standardize.canonical(stream, onset_ms, end_ms)` and the window functions re-canonicalise a stream on
  detected bounds;
- `screen.wrong_hand_flags(...)` is the D6 screen for use inside a fold;
- `plotting.figure` and `plotting.save_fig(fig, name, caption=..., decision=...)` produce every figure.

**Rules for lanes:**
- group by `person_id` only;
- never train on natural-use trials or return windows;
- evaluate only through `run_experiment`, with a unique `name`;
- keep seeds fixed (`config.SEED`, `fold.seed`);
- install no packages;
- copy no data out of `analysis/data/`;
- name no person anywhere;
- make figures with `plotting`, so they share the style and the caption registry.

## 10. Tests (`stage2/tests`)

Snapshot counts are pinned in ONE file, `stage2/tests/expected_counts.json` (fixture `expected`); its
`_note` and section 11 say how to update it when a participant is added.

- **Curation**: the counts, the multi-session persons, the name-free reason codes, the refusal of the
  private file, the shared code (P01 and P13) and the split code (P14), and per-session demographics.
- **Labels**: every primary label is instructed and equals the hand of its planned block in the raw
  session records; the natural-use set is exactly the accepted reported trials, one first trial per person.
- **Exclusion cascade**: the flow counts, first-failing-rule bookkeeping and order sensitivity; flags and gates
  never exclude.
- **CV harness**: no person overlap, by person rather than by code; no label columns on the test side; a
  scaler fitted on test data is detected in any fold; the metrics match sklearn; ECE, bootstrap, the sign-flip
  test, k-attempt curves on the same persons, natural-use rows, the D2 cap, gated metrics and the result round trip.
- **Standardisation**: shapes and references, exact mirror covariance, screen-flip invariance of the
  gravity channels, the same code on detected bounds, phases, and cache alignment.
- **Cue-free detection**: synthetic streams, the evaluation contract and the stress test.
- **Privacy**: no participant name token (case-insensitive) in any output or source file, figure texts and PDFs
  included; no quasi-identifier in a report-facing table.
- **Spectra and provenance** (`test_qc_provenance.py`): no FFT crops a window; the cache and every result name
  their code and data identity.
- **Label screen** (`test_screen.py`): the swap signature and the D6 screen.
- **Pipeline** (`test_pipeline.py`): discovery, ORDER and DEPENDS, the step contract, a failing step stops the
  run, selection, staleness helpers, the lane naming conventions, the reference worked example.
- **Describe** (`test_describe.py`): every figure function draws with the cache and the raw data locked away,
  captions take their counts from the data, sections map to figures, the saved inputs are current, the numbers
  file covers the cited values.
- **Locking** (`test_locking.py`): the lock is exclusive and released, stale locks are broken and fresh ones are
  not, atomic replacement, `update_registry` writes the same bytes as before, and parallel processes lose no
  registry row and no caption (without the lock, 4 processes writing 25 rows each kept 31–36 of 100).
- **Report export** (`test_report_export.py`): booktabs tables, value formats, macro names, numbers round trip,
  report-safe tables, collecting figures and tables.
- **Exploratory guard** (`test_exploratory_guard.py`): no exploratory person, session or app code in the main
  snapshot, curation or cache. The tests that read the exploratory persons' files (three here, three in
  `test_x1.py`) carry the marker `exploratory` and are skipped unless selected, so ANY lane can run the whole
  suite without reading them; the orchestrator (or lane x1) runs them with
  `python -m pytest stage2/tests -m exploratory -q -p no:cacheprovider`, or the whole suite with
  `STAGE2_EXPLORATORY_TESTS=1`. A default check needs no exploratory file: every main person id has the main form.
- **Core maintenance 2026-10-28** (`test_cv.py`, `test_pipeline.py`, `test_report_export.py`, `test_p1.py`,
  `test_cuefree_privacy.py`): the vectorised `paired_compare`, the sign-flip test by doubling and the numpy ECE
  equal the former code (kept in the tests as references; on saved results); the tie rule everywhere; the
  re-score is exact and idempotent; declared specifications and input fingerprints; `--threads` wins through the
  real command line; the pipeline log under its lock; the table-kind guard, header-only tables, atomic writes,
  caption underscores, `save_fig`; no result number typed into a core docstring.
- **Pinned counts** (`test_expected_counts.py`): the fixture equals the cache.

## 11. Pipeline and lane conventions

Every result, figure, table and number is regenerated by one orchestrator, `stage2/pipeline.py`. Compute and
presentation are separate: a **compute** step runs experiments and saves everything the presentation needs
(predictions, per-fold information, curves, importances, the data behind each figure) to `results/` and
`tables/`; the **figures**, **tables** and **numbers** steps only read those saved files and never fit a
model, so redrawing a figure after changing an axis takes seconds.

```
cd analysis
.venv/Scripts/python.exe -m stage2.pipeline --list                       # units, their steps, cache status
.venv/Scripts/python.exe -m stage2.pipeline all                          # whatever is missing or stale
.venv/Scripts/python.exe -m stage2.pipeline all --force                  # rebuild everything from the snapshot
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m2           # one lane's missing / stale experiments
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m2 --only "m2_cnn*" --force
.venv/Scripts/python.exe -m stage2.pipeline figures                      # redraw every figure from saved results
.venv/Scripts/python.exe -m stage2.pipeline figures --lanes m2 --only M2_03_k_curve
.venv/Scripts/python.exe -m stage2.pipeline tables                       # CSV + LaTeX booktabs, registry_all.csv
.venv/Scripts/python.exe -m stage2.pipeline numbers                      # <unit>_numbers.json -> results/numbers.tex
.venv/Scripts/python.exe -m stage2.pipeline report                       # asset step + latexmk -> report/main.pdf
.venv/Scripts/python.exe -m stage2.pipeline report --no-latex            # asset step only
```

| Target | What runs |
|---|---|
| `cache` | builds the cache when it is missing or **stale**; `--force` rebuilds it anyway |
| `describe`, `reference` | the core unit of that name: all its steps |
| `compute` | every selected unit's compute step; experiments whose result is missing or stale (`--force`: all; `--only`: these names). Ensures the cache first but never force-rebuilds it |
| `figures`, `tables`, `numbers` | that step of every selected unit; `tables` then merges the registries into `results/registry_all.csv`, `numbers` writes every macro to `results/numbers.tex` |
| `report` | the LaTeX report: `report/scripts/make_assets.py` copies every figure listed in a `captions.json`, every `.tex` table and every numbers file into `report/` (see report/README.md), then latexmk builds `report/main.pdf` (`--no-latex`: the asset step only; `REPORT_MODE=final`: the final-mode guards) |
| `all` | cache, then compute, figures, tables, numbers of every unit (stage by stage), then report |

**Units.** Core units are modules under `stage2/scripts`: `cache` (`build_cache.py`, ORDER 0), `describe`
(`describe.py` + `describe_figures.py`, ORDER 10) and `reference` (`reference.py` + `reference_figures.py`, ORDER
20). Lanes are discovered from `stage2/lanes/<id>_pipeline.py`. A unit module defines `STEPS = {'compute': fn,
'figures': fn, 'tables': fn, 'numbers': fn}` (any subset), `ORDER` (core 0-20, model lanes 100-199, exploratory
500, synthesis 900: a synthesis lane runs after the model lanes) and optionally `DEPENDS = ('m1', ...)` (ordering
only; lanes stay runnable alone). Every step is `fn(force: bool, only: list[str] | None) -> None`. Each step
logs its duration (also to `reports/stage2/pipeline_log.jsonl`); the first failure stops the run with the unit,
the step and the traceback (exit code 1).

**Staleness.** The cache records its *data identity* (`meta.json` `data_identity`: snapshot sha256, curation
sha256, cache parameters, builder code = `data.py`, `standardize.py`, motionlab; `data.data_identity()`), and
every result records the `data_id` of the cache it was made from (`info.cache_code.data_id`).
`pipeline.cache_status()` says whether the cache matches its inputs; `pipeline.result_current(name)` whether a
result was made from the current cache; derived JSON files written with `Lane.write_json` carry the same
`provenance` (`pipeline.file_current(path)`). So after a participant is added, a plain `all` rebuilds the cache
and recomputes every result. Editing presentation code (`plotting.py`, a figures module) or the harness
(`cv.py`, `pipeline.py`) makes nothing stale. `compute --force` recomputes experiments but never rebuilds the
cache; only `cache --force` and `all --force` do. **Never rebuild the cache while lanes are running** (they read
it).

**Specification and input fingerprints (opt-in).** The data identity alone does not see an edited experiment
specification or analysis code. A lane can ask for more:
- `L.should_run(name, force, only, spec={...})` (and `pipeline.result_current(..., spec=...)`) also recomputes a
  result whose declared specification differs from the lane's current one. The declared specification of a result
  (`cv.result_spec`: model, params, notes, n_configs_tried, sets, cap_per_person, seed, n_boot, leak_check,
  test_meta_columns, plus the lane's own `spec=` extras of `run_experiment`, e.g. a code version) is read from fields
  every saved result already stores, so existing results need no rewrite and none became stale; new results also
  store its sha256 (`info.spec_sha256`). A code version: `v = pipeline.source_fingerprint(m9_models)` (module
  objects, dotted names or paths), passed as `run_experiment(..., spec={'code': v})` and
  `L.should_run(name, force, only, spec={'code': v, 'params': params})`; fingerprint only the modules that define
  the model (any edit of them, a comment included, recomputes the result).
- `L.write_json(name, obj, inputs=[...])` records the sha256 of the files a derived file is computed from (result
  names, output-relative paths, module objects or source paths relative to analysis/) in `provenance_inputs`;
  `file_current` then also requires them unchanged. Files without that record keep the data-identity rule.

No existing lane opts in, so a plain `compute` reruns none of their saved results
(`test_every_saved_result_is_current_so_compute_is_a_no_op`, and a real `python -m stage2.pipeline compute` after
the 2026-10-28 maintenance reran no experiment). Without opting in, an edited spec or analysis function still
needs `compute --only <names> --force`; lanes m3 (inputs signature) and x1 (manifest) keep their own finer checks.

**Core describe.** `describe.py` is split like a lane. Its compute step runs the analysis sections
(`describe.SECTIONS`: participants, exclusion flow, completion, conditions, raw sampling and spectra, durations
and segmentation, example traces, class means, sign patterns, first-half yaw, platforms, return, cue-free v0 and
its stress test, closure, flags and gates, label consistency; about 45 s on one free core) and saves the
descriptive tables `tables/*.csv`, `tables/describe_summary.json` and the figure inputs
`tables/describe/describe_*.json|csv` (with provenance); it is skipped when they were made from the current cache.
`compute --lanes describe --only <section or figure name> --force` reruns only the sections named (or the
section that feeds the figure named) and updates their summary keys in place. The figures step
(`describe_figures.py`: one function per figure, `FIGURES` holds every presentation default) draws S2_01..S2_20
from those files in seconds and writes the plotted data to `tables/describe/<figure>.csv`; the tables step writes
twelve booktabs tables `tables/describe/describe_<name>.csv|.tex`; the numbers step writes
`results/describe_numbers.json` (about 160 macros `\Describe<Key>`).

**Shared files are process-safe.** Lanes run in parallel, and two kinds of file are read-modify-written by
several processes: the results registries (`cv.update_registry`) and the figure registries `captions.json` /
`figure_texts.json` of a folder (`plotting.save_fig`). Both hold a cross-process lock (`<file>.lock`, created with
`os.open(O_CREAT | O_EXCL)`, retried with exponential backoff, broken when older than 120 s because its owner
died) and replace the file atomically (`locking.atomic_path`), so no update is lost and no reader sees a
half-written file. Lanes still keep their own registry (`results/registry_<id>.csv`) and figure folder. Since
2026-10-28 also: `pipeline_log.jsonl` is appended under its lock; every result, table (`.tex` first, then
`.csv`), numbers file, `numbers.tex`, derived JSON, finalists file and `registry_all.csv` is replaced atomically;
and `save_fig` renders to temporary files and moves the images into place together with their registry entries
under the folder's lock, so a reader holding that lock never sees a PDF without its entries.

**Table kinds.** `tables/<id>/` holds report tables (CSV + `.tex`, `Lane.export_table`) and derived or figure
data (a CSV only: `export_table(..., tex=False)`, `Lane.figure_data`). `report_export.check_table_kind` refuses to
let one kind overwrite the other's CSV (`TableNameCollision`): give derived data its own name (e.g.
`<name>_rows`). A derived JSON may share a report table's stem (no file is overwritten). A table needs columns: an
empty one is written header-only (`pd.DataFrame(rows, columns=[...])`), a columnless one is refused. Captions and
notes are LaTeX; a bare `%`, `_`, `&`, `#` or `^` (outside math and outside `\ref`, `\label`, `\cite`, `\url`,
...) is escaped. Every CSV goes through `motionlab.report.write_csv`, which puts an apostrophe before a TEXT cell starting
with `+`, `-`, `=` or `@` (a spreadsheet would run it as a formula): a signed estimate string of a report table reads
`'+2.9 [...]` in its CSV, by design; numeric columns are untouched, and the `.tex` and the numbers files carry the
plain values.

### Per-lane files (lane id `<id>`, lower case, e.g. `m2`; figure prefix `<ID>`, e.g. `M2`)

| What | Where | Written with |
|---|---|---|
| lane code | `stage2/lanes/<id>_*.py` | |
| pipeline module | `stage2/lanes/<id>_pipeline.py`: `STEPS`, `ORDER`, `DEPENDS` | |
| figure functions | `stage2/lanes/<id>_figures.py`: one function per figure, named after it; presentation defaults in a `FIGURES` dict at the top | |
| tests | `stage2/tests/test_<id>*.py` | |
| experiment results | `results/<id>_<name>.json` | `Lane.run_experiment` (= `cv.run_experiment`, name checked) |
| registry | `results/registry_<id>.csv` (one per lane; merged into `registry_all.csv`) | `Lane.run_experiment` |
| derived data (curves, importances, figure inputs) | `tables/<id>/<id>_<name>.json` or `.csv` | `Lane.write_json`, `Lane.export_table(..., tex=False)` |
| figures | `figures/<id>/<ID>_NN_short_name.pdf|png`, `captions.json`, `figure_texts.json` | `Lane.save_fig(fig, name, caption=..., decision=..., data=df)` |
| figure data | `tables/<id>/<ID>_NN_short_name.csv` (tidy: what is plotted) | `Lane.save_fig(..., data=df)` / `Lane.figure_data` |
| report tables | `tables/<id>/<id>_<name>.csv` + `.tex` (booktabs) | `Lane.export_table` (= `report_export.export_table`) |
| numbers | `results/<id>_numbers.json`: `{key: {value, format, source, description}}` | `Lane.write_numbers`, `report_export.numbers_entry` |
| finalists | `results/finalists_<id>.json`: `[{name, results_json, module, build_x, factory, params, n_configs_tried, notes}]` | `Lane.write_finalists` |
| summary | `reports/stage2/<id>_SUMMARY.md` | `Lane.summary_path` |

Numbers become LaTeX macros `\<Unit><Key>`: the unit and the lower_snake_case key in CamelCase with digits
spelled out (`m2` / `acc_k3` -> `\MTwoAccKThree`; `reference` / `accuracy_lo` -> `\ReferenceAccuracyLo`). Formats
are Python format specs: `'.1%'` (x 100, `\%` in LaTeX), `'.2f'`, `'d'`, `'+.1%'`, `''` for text. Every number
the report text cites must come from a numbers file; `source` names the saved file it was read from.

Report tables and figure data go through `report_export.report_safe`: `trial_id` becomes `trial_key`,
`session_id` becomes `session_label`, and an app code, backend id or timestamp is refused. The privacy tests
scan every lane folder too.

### Template: `stage2/lanes/m9_pipeline.py`

```python
"""Lane m9: <one line>. Steps: compute (experiments + derived data), figures, tables, numbers."""
from stage2.pipeline import Lane, limit_threads

limit_threads(2)                                   # this lane's thread budget, BEFORE numpy loads

import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from stage2 import report_export as RX            # noqa: E402

L = Lane('m9')
ORDER = 190
DEPENDS = ()
EXPERIMENTS = ('m9_yaw_logreg',)


def build(d):
    """name -> (fit_predict, X, notes, n_configs_tried); tuning only inside fit_predict on fold.inner_splits()."""
    from stage2.lanes.m9_models import yaw_logreg
    return {'m9_yaw_logreg': (yaw_logreg(d.channels), d.approach, 'pre-specified', 1)}


def compute(force=False, only=None):
    todo = [n for n in EXPERIMENTS if L.should_run(n, force, only)]
    if not todo:
        print('m9: results current; skipped (--force recomputes)')
        return
    from stage2.data import load_cache
    d, nat = load_cache(), load_cache(('natural_use',))
    specs = build(d)
    for name in todo:
        fp, X, notes, k = specs[name]
        L.run_experiment(name, fp, X, d.trials, extra=(nat.approach, nat.trials), notes=notes, n_configs_tried=k)
    # anything else a figure needs, saved now (never recomputed by the figures step):
    r = L.load_result('m9_yaw_logreg')
    L.write_json('m9_person_acc', {'per_person': r.per_person.to_dict(orient='records')})


def figures(force=False, only=None):
    from stage2.lanes import m9_figures
    m9_figures.draw(only)


def tables(force=False, only=None):
    r = L.load_result('m9_yaw_logreg')
    df = pd.DataFrame([{'model': 'yaw logreg', 'accuracy': r.metrics['accuracy'], 'auc': r.metrics['roc_auc']}])
    L.export_table(df, 'm9_models', caption='Lane m9 models (leave-one-person-out).',
                   column_formats={'accuracy': '.1%', 'auc': '.3f'})


def numbers(force=False, only=None):
    E = RX.numbers_entry
    r = L.load_result('m9_yaw_logreg')
    src = 'results/m9_yaw_logreg.json'
    L.write_numbers({'accuracy': E(r.metrics['accuracy'], '.1%', src, 'LOSO accuracy'),
                     'accuracy_lo': E(r.ci['accuracy'][0], '.1%', src, '95% CI low'),
                     'accuracy_hi': E(r.ci['accuracy'][1], '.1%', src, '95% CI high')})


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
```

### Template: `stage2/lanes/m9_figures.py`

```python
"""Figures of lane m9, from saved files only. Edit FIGURES, then: python -m stage2.pipeline figures --lanes m9"""
import pandas as pd

from stage2.pipeline import Lane, selected
from stage2.plotting import HAND_COLORS, figure

L = Lane('m9')
FIGURES = {
    'M9_01_person_accuracy': dict(width='half', height=2.2, xlim=(0, 1), xticks=(0, 0.5, 0.8, 1.0),
                                  xlabel='LOSO accuracy', color=HAND_COLORS['right']),
}


def fig_M9_01_person_accuracy(**kw):
    o = {**FIGURES['M9_01_person_accuracy'], **kw}
    pp = pd.DataFrame(L.read_json('m9_person_acc')['per_person']).sort_values('accuracy')
    fig, ax = figure(o['width'], o['height'])
    ax.barh(range(len(pp)), pp['accuracy'], color=o['color'])
    ax.set_yticks(range(len(pp)), pp['person_id'])
    ax.set_xlim(*o['xlim'])
    ax.set_xticks(o['xticks'])
    ax.set_xlabel(o['xlabel'])
    n = len(pp)                                     # computed, never typed: captions carry no data-specific count
    L.save_fig(fig, 'M9_01_person_accuracy', data=pp[['person_id', 'accuracy']],
               caption=f'Leave-one-person-out accuracy of each of the {n} persons (lane m9).',
               decision='Which persons the model fails on.')


FIGURE_FUNCTIONS = {'M9_01_person_accuracy': fig_M9_01_person_accuracy}


def draw(only=None):
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
```

`stage2/scripts/reference.py` and `reference_figures.py` are the full worked example (experiments, a derived
summary with paired comparisons, one two-panel figure with its config dict and figure data, two booktabs
tables, a numbers file); `describe.py` and `describe_figures.py` show the same contract for a large
descriptive unit (analysis sections, saved figure inputs, twenty figure functions, twelve tables).

### Rules

- Hard-code nothing data-specific (person, trial or fold counts, person ids, result numbers) in code,
  captions or text: compute it from the data or the saved results. An unavoidable choice (for example a
  sensitivity analysis on particular blocks) is data-driven (`reference.swap_blocks`) or configured in one
  clearly named constant.
- Deterministic and idempotent: seeds from `config.SEED` / `fold.seed`; tables and numbers carry no
  timestamp; the same inputs give the same files.
- Names: results and tables `<id>_*`, figures `<ID>_NN_short_name`; `Lane` checks them.
- Threads: `limit_threads(n)` at the top of the lane's pipeline module, `n_jobs <= n`, `torch.set_num_threads(n)`.
  `--threads N` on the pipeline command line wins over every `limit_threads(n)` of that process (the lanes' own
  budgets included); worker processes a lane starts keep their own budget. On a shared machine, several
  one-thread worker processes run tree boosting much faster than one multi-thread process (OpenMP spin-waiting).
- Expensive nested tuning: the default leak check (`leak_check='all'`) re-runs `fit_predict` on half of the test
  rows in every fold. A lane may cache the tuned fit keyed by its configuration and a hash of the TRAINING rows
  (the fit never sees `X_test`), so the check still exercises the prediction step without a second tuning
  (lane m2 does).
- Lanes run alone (`--lanes <id>`) and in parallel with other lanes: never `--force` the cache, never write
  another lane's files, never the shared `results/registry.csv` (use the lane registry).

### Adding a participant

1. Put the new session in the snapshot (backend layout `nose-touch-v1/<code>/<session>/`), or export a new
   snapshot and point `config.SNAPSHOT_NAME` at it.
2. Add its row to `analysis/data/curation/sessions.json`: `{"session_id": "...", "person_id": "P21",
   "include": true}`.
3. `python -m stage2.pipeline all --force` (a plain `all` also works: the cache's data identity changed, so it
   is rebuilt and every result, the describe outputs included, recomputed).
4. `python -m stage2.scripts.expected_counts --without P21` must reproduce the pinned counts exactly (every
   change is explained by the new person); then `python -m stage2.scripts.expected_counts --write`.
5. `python -m pytest tests stage2/tests -q -p no:cacheprovider`.

Exploratory persons (PD01, ...) never enter the snapshot or `sessions.json`: they live in
`analysis/data/exploratory/` and `curation/exploratory.json` (`config.EXPLORATORY_*`), only the exploratory
lane x1 reads them, and `tests/test_exploratory_guard.py` checks that none of their ids reaches the main
snapshot, curation or cache.

## 12. Provenance and privacy

Every output can be traced to the data and the code that made it, without a timestamp in any report-facing file:

| Output | Records |
|---|---|
| cache `meta.json` | cache version, creation time, snapshot name and sha256 over every file, git commit and dirty flag, code fingerprint (sha256 of `stage2/*.py`, `stage2/scripts`, motionlab, `requirements-stage2.txt`), every parameter, curation sha256, the data identity (`data_id` over snapshot, curation, parameters and builder code), channel table, flag and gate definitions, counts |
| `results/<name>.json` (`info`) | creation time, git commit and dirty flag, code fingerprint, cache version, the cache's code, snapshot sha256, creation time and `data_id`, seed, bootstrap size, `n_configs_tried`, the D2 cap and the rows it dropped, the leak check per fold, the test-side columns, `locked_test: waived` |
| registry rows | git commit, dirty flag, stage2 code hash prefix, cache version, `n_configs_tried`, leak check |
| derived JSON (`describe_summary.json`, `tables/describe/describe_*.json`, `reference_summary.json`, `Lane.write_json`) | `provenance`: `data_id`, cache version, snapshot sha256 (`pipeline.file_current` compares it) |
| numbers | `source`: the saved file each value was read from |
| LaTeX tables | a "generated by" comment naming the command that rebuilds them |
| figures | caption and decision in `captions.json`, every drawn string in `figure_texts.json` |

**Privacy.** No name-bearing field reaches an output: the curation's free-text reasons are replaced by fixed
codes and its `rules` field is never read, the private name file is never opened (`config.assert_not_private`),
the session table drops `user_agent`, and the raw `nickname` is never copied. Report-facing tables and figure data
carry person ids (P01..P20), pseudonymous session labels (`P14-s3`) and trial keys (`P14-s3-t07a1`) only:
`report_export.report_safe` (and `describe.write_table`) refuse app codes, backend ids, timestamps, OS and browser
versions and screen geometry (`data.QUASI_IDENTIFIER_COLUMNS`). Tests scan every table, JSON, LaTeX file, caption,
figure text, PNG and (with pdftotext) PDF, the cache tables and every source file for the name tokens of the raw
records, case-insensitively (`test_cuefree_privacy.py`). The cache keeps the backend keys (git-ignored) for tracing.

## 13. Audit findings and where they are handled

| # | Finding | Handled by | Checked by |
|---|---|---|---|
| 1 | harness leaks: grouping by app code or session, label columns in X, trial_id lookups on the test side, natural-use rows as training rows | `cv.run_experiment` checks 1, 2, 5, 6, 7 and 12 of section 5.1 | `test_grouping_by_code_or_session_is_refused`, `test_label_in_X_is_refused`, `test_test_side_has_no_lookup_key_and_the_cache_is_locked`, `test_natural_use_and_excluded_rows_need_an_explicit_opt_in` |
| 2 | the leak check ran on fold 0 only | every fold by default (`config.LEAK_CHECK = 'all'`), recorded per fold | `test_a_leak_in_a_later_fold_is_caught_by_default`, `test_sampled_leak_check_records_its_folds` |
| 3 | missing flag columns | `data.FLAG_COLUMNS` (section 2.1) | `test_flags_are_never_exclusions_and_gates_are_scored_apart` |
| 4 | P08's first two blocks look hand-swapped | the swap signature found from the data (`screen.block_consistency`), labels never changed, sensitivity analyses in `reference` (section 5.5); no typed person or block anywhere | `test_swap_signature_*`, `test_reference_outputs_follow_the_contract`, `test_no_typed_counts_in_captions` |
| 5 | undocumented deviations from docs/09 | section 5.4; `info.locked_test`, `n_configs_tried`, the cap record in every result | this README |
| 6 | the spectrum FFT truncated long windows | `qc.fft_length`, `qc.cumulative_power` (refuses a window longer than the FFT) | `test_long_window_is_not_cropped`, `test_spectrum_summary_uses_the_full_windows` |
| 7 | the k-attempt curve used other persons at k = 5 | `cv.k_attempt_curve(common_persons=True)`: the same persons at every k | `test_k_attempt_curve_uses_the_same_persons_for_every_k`, `test_reference_k_curve_reports_its_persons` |
| 8 | the cue-free stillness statistic is optimistic | section 4: the caveat and the stress evaluation | `test_stress_shortened_stillness_breaks_v0_below_its_rule`, `test_evaluation_on_the_cache` |
| 9 | the privacy test was case-sensitive and skipped figures | case-insensitive scan of every output, figure texts, PNGs and PDFs | `test_no_participant_name_in_any_output_or_source`, `test_figure_texts_are_recorded_and_the_pdfs_are_scanned` |
| 10 | the cache had no code fingerprint | `meta.json` `code` and `data_identity` (section 12) | `test_cache_and_results_carry_the_code_fingerprint` |
| 11 | quasi-identifiers in report-facing tables | `report_export.report_safe`, `describe.write_table`, `data.report_sessions` | `test_report_tables_carry_no_quasi_identifiers`, `test_report_safe_refuses_quasi_identifiers` |
| 12 | the natural-use headline must use one first free trial per person | rule N1 (section 5.3) | `test_natural_use_headline_is_one_first_trial_per_person` |
| 13 | a P(right) of exactly 0.5 (an abstention, a constant predictor) scored as a right-hand answer, so every abstention was correct on the all-right natural-use set | half credit for a tie in every accuracy (`cv.credit`, section 5.1 item 14); `coverage` and `accuracy_answered` when a model abstains; saved results re-scored from their predictions (`stage2.scripts.rescore`, list in `maintenance/rescore_ties.csv`) | `test_a_tie_earns_half_credit_everywhere`, `test_k_attempt_curve_gives_half_credit_to_a_mean_of_one_half`, `test_rescore_leaves_tie_free_results_alone_and_is_idempotent` |
| 14 | `paired_compare` about 50 times slower than necessary (two pandas lookups per resample) | vectorised on the same random stream; exact sign-flip test by doubling | `test_vectorised_paired_compare_equals_the_former_loop_on_saved_results`, `test_sign_flip_by_doubling_equals_the_enumeration` |
| 15 | the ECE built a DataFrame per bootstrap resample | numpy ECE on the bins of `reliability` (bit-identical) | `test_numpy_ece_equals_the_reliability_formula` |
| 16 | `--threads N` silently overridden by a lane module's own `limit_threads(n)` | `pipeline.set_thread_override` before the lanes are imported | `test_threads_on_the_command_line_win_over_the_lane_budget`, `test_thread_override_is_process_local` |
| 17 | derived data and report tables overwrote each other in `tables/<id>/` | `report_export.check_table_kind` (`TableNameCollision`), section 11 | `test_a_report_table_and_derived_data_never_overwrite_each_other` |
| 18 | unlocked `pipeline_log.jsonl`, tables and numbers written in place, a PDF visible before its registry entry | the log appended under its lock; atomic writes; `save_fig` moves the images in with their entries under the folder's lock | `test_the_pipeline_log_is_appended_under_its_lock`, `test_tables_and_numbers_are_replaced_atomically`, `test_save_fig_moves_the_images_in_with_their_registry_entries` |
| 19 | an empty table written as an unreadable 2-byte CSV | an empty table is written header-only, a columnless one is refused | `test_an_empty_table_is_written_header_only_and_a_columnless_one_is_refused`, `test_report_tables_carry_no_quasi_identifiers` |
| 20 | a bare `_` (or `^`, `&`, `#`) in a caption stopped pdflatex | escaped outside math and reference arguments; every report table compiles with the report's table packages | `test_bare_underscores_in_captions_are_escaped_outside_math_and_references`, `test_bare_specials_in_captions_are_escaped_outside_math` |
| 21 | result numbers typed into core docstrings went stale (`cuefree.py`) | pointers to the regenerated files | `test_core_docstrings_type_no_result_numbers` |
| 22 | staleness saw only the cache's data identity | opt-in declared specifications and input fingerprints (section 11) | `test_opt_in_spec_and_input_staleness`, `test_every_saved_result_is_current_so_compute_is_a_no_op` |
| 23 | the exploratory guard test read a file only lane x1 and the orchestrator may read | marker `exploratory`, run only when selected (section 10) | `test_exploratory_tests_run_only_when_selected` |
| 24 | the percentile CI of the ECE is biased upwards and can exclude its estimate | documented (section 5.1 item 15): judge an ECE against its perfect-calibration null or through paired differences | lanes m1, m3, m4 (`*_stats.ece_null`) |

**Deferred to the next planned cache rebuild** (each needs an edit of `data.py` or `standardize.py`, which changes the
cache's data identity and so would mark every result stale): the `data.py` docstrings still list the persons as
"P01..P19" (stale since a participant was added; nothing reads them); `Stage2Data.stream(i)` does not carry the session's calibrated gravity magnitude (lane x1 recovers it per
trial from the cached fixed window).
