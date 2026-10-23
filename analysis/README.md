# motionlab: the Stage-2 analysis starter kit

`motionlab` loads NoseTouch recordings (folder trees from the backend or the local receiver, and export
bundles shared from the app), checks their quality, normalises platform conventions, re-segments the
movement, computes grip-invariant and chiral features, and evaluates left/right classifiers with
subject-wise cross-validation. Everything is plain Python (numpy, scipy, pandas, scikit-learn,
matplotlib, jsonschema); there is nothing to compile.

```
analysis/
  motionlab/          the package (import it from analysis/, or add analysis/ to sys.path)
    schema.py         validation against ../schema/records.schema.json (the shared contract)
    io.py             load_dataset(): folder trees, export bundles, zips, envelopes; dedupe; tidy tables
    conventions.py    frames, units, norm_v1 (gravity sign, rotationRate mapping), orientation tables
    preprocess.py     resampling, zero-phase filters, gravity_v1 (gyro propagation), Chrome rounding
    segment.py        online_v1 (exact port of the app's detector) and offline_v1
    features.py       basic / device / invariant / chiral features, sequence tensors
    augment.py        mirror, grip rotation, time warp, magnitude scaling, quantisation, noise
    models.py         YawSignRule, logistic regression, gradient boosting, the model registry
    evaluate.py       CV schemes, metrics, participant-bootstrap CIs, subgroup table, mirror consistency
    qa.py             dataset quality summary
    synth.py          an independent Python generator of schema-valid records (tests, demos)
    plots.py, report.py   figures (matplotlib Agg), Markdown and formula-safe CSV helpers (write_csv)
  scripts/            qa_report.py, train_baseline.py, plot_trial.py (each has --help)
  tests/              pytest suite (+ fixtures/ with reference runs of the app's own TypeScript code)
  data/               your data, git-ignored (see data/README.md)
  reports/            script output, git-ignored (see reports/README.md)
```

## Setup

Python 3.10 or newer (CI uses 3.11).

**Windows (PowerShell)**

```powershell
cd analysis
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # if scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python -m pytest tests -q
```

Without activating, call the interpreter directly: `.venv\Scripts\python.exe scripts\qa_report.py data\synthetic`.

**macOS / Linux**

```bash
cd analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

`requirements.txt` lists minimum versions and, in its header, the versions the kit was tested with.
Freeze the environment of any reported result: `pip freeze > requirements.lock.txt`.

## Data layout

Put data under `analysis/data/` (git-ignored, see [data/README.md](data/README.md)). Every loader
accepts any mix of:

| Input | What it looks like |
|---|---|
| Folder tree (Apps Script Drive folder, `server/received/`, `npm run synth`) | `<root>/<study_code>/<participant_id>/<session_id>/session.json` + `trial_<idx:3>_a<attempt>_<trial_id>.json` |
| Export bundle from the app's Data screen | one JSON `{"format": "nose-touch-export", participants, sessions, trials}`; also `*.json.txt` (Chrome on Android re-shares it as text) |
| Archives | `*.zip` (a Drive folder download), `*.json.gz`, `*.jsonl` |
| Upload envelopes, arrays of records | `{kind, study_key, sent_at, record}` |

Directories are walked recursively and files are classified by content, not by name. Records are
validated against `schema/records.schema.json` (invalid ones are skipped and listed in
`Dataset.issues`) and deduplicated (docs/02 section 10): copies with the same id keep the newest
`updated_at` / `created_at` and are listed in `Dataset.duplicates`; two different trial ids with the
same `(session_id, trial_index, attempt)`, and several accepted attempts of one planned trial, are kept
and flagged (`dup_attempt_key`, `multi_accepted`).

## Commands

From `analysis/` with the venv active:

```bash
# synthetic data with the app's real pipeline (from app/):  npm run synth -- --participants 12 --preset standard --out ../analysis/data/synthetic
python scripts/qa_report.py data/synthetic                       # -> reports/qa_synthetic/qa_report.md
python scripts/train_baseline.py data/synthetic                  # -> reports/baseline_synthetic/report.md
python scripts/plot_trial.py data/synthetic --list 10            # trial ids to choose from
python scripts/plot_trial.py data/synthetic --trial 16cf69f9     # id or unique prefix -> reports/trials/
```

| Script | Writes | Main options |
|---|---|---|
| `qa_report.py <data...>` | `qa_report.md`, `tables/*.csv`, `summary.json`, sampling / count / duration figures | `--out DIR`, `--simulated include\|exclude\|only`, `--no-online-check`, `--no-offline`, `--sample N` |
| `train_baseline.py <data...>` | `report.md`, `metrics.csv`, `subgroups.csv`, `mirror_consistency.csv`, `oof_<scheme>.csv`, `participant_accuracy.csv`, `exclusions.csv`, `features.csv`, figures | `--out DIR`, `--models`, `--schemes`, `--n-splits`, `--n-boot`, `--augment-mirror`, `--quantize none\|ios\|all`, `--segment`, `--simulated`, `--require-calibration` |
| `plot_trial.py <data...> --trial ID` | `trial_<id>.png` | `--trial` (repeatable, prefix ok), `--list N`, `--out DIR` |

The QA report covers participants and conditions, duplicates, sampling rate / gaps / repeated samples /
clipping per platform, the gravity-sign source and rotationRate mapping per session (and every
calibrated sign the trial votes overrode), a gyro-gravity closure check, segmentation success (stored
online_v1, this port's agreement with it, offline_v1) and orientation mismatches. For real data use
`--simulated exclude`.

The CSV files carry strings from the records unchanged (warnings, notes, phone models, ids), and anyone
can post schema-valid records to the public endpoint. Text cells that Excel, Google Sheets or LibreOffice
would run as a formula (starting with `=`, `+`, `-`, `@`, a tab or a line break) are therefore written
with a leading single quote (`motionlab.report.write_csv`, used for every CSV the kit writes). Still keep
only the intended `study_code` and `simulated == false` records before reporting.

`train_baseline.py` applies the exclusion rules T1-T7 of docs/02 section 12, computes the features of
every trial and of its mirror image, and runs every model on the same subject-disjoint splits of each
scheme:

| Scheme | Splits | Question |
|---|---|---|
| `stratified_group_kfold` | 5 participant folds, stratified on EHI class x platform, seeded (the development CV) | new people |
| `loso` | leave one participant out | per-person accuracy distribution |
| `orientation` | one held-out orientation (mirror pairs `top_left_*`/`top_right_*` together) x participant folds | a new grip and new people |
| `platform` | train on one platform, test on the other (people seen on both are dropped from training) | iOS <-> Android transfer |
| `group_kfold` | plain GroupKFold | |

The report gives accuracy, balanced accuracy, ROC AUC (95 % participant-bootstrap CIs), log loss, Brier,
per-participant accuracy, the mirror consistency of every model, a label-permutation leakage control and
the subgroup error table (sex, handedness, EHI class, age band, orientation, speed, start position,
variant, platform, hand, dominant hand) with participant-bootstrap CIs. Synthetic data give near-perfect
numbers by construction: they test the pipeline, never the science.

## Python API

```python
from motionlab import load_dataset
from motionlab.features import extract_features
from motionlab.evaluate import analysis_set, modeling_table, evaluate_models, summary_table
from motionlab.models import MODEL_REGISTRY

ds = load_dataset('data/synthetic')                 # ds.trials / ds.sessions / ds.participants / ds.issues
keep, exclusions = analysis_set(ds.trials, simulated='include')
F, F_mirror = extract_features(ds, ds.trials.loc[keep, 'trial_id'], mirror='both')
X, meta = modeling_table(F, ds.trials)
res = evaluate_models({k: MODEL_REGISTRY[k] for k in ('yaw_rule', 'logreg', 'hgb')}, X, meta,
                      scheme='stratified_group_kfold', X_mirror=F_mirror.reindex(X.index), n_boot=1000)
print(summary_table(res))

sig = ds.signals(ds.trial_ids[0])                   # normalised device-frame acc / accg / gyro (norm_v1)
```

## Normalisation rules (`norm_v1`)

Raw records are never modified; `conventions.session_normalization` resolves one `Normalization` per
session and every feature is computed after it (`Dataset.signals`, `features.prepare_trial`). The
version string `norm_v1` is stored with every feature row (`qc_norm_version`). Spec: docs/03 section 5
and docs/05 section 2.1.

**Device frame (W3C):** x = right edge, y = top edge, z = out of the screen, fixed to the device.

**Rotation rate.** `rotationRate.alpha`, `beta`, `gamma` are the rates about device **x, y, z** (W3C CRD
12 Feb 2025; MDN's "alpha = Z" is outdated), so the angular velocity is `w = [alpha, beta, gamma]`
deg/s. The calibration tilt check overrides this per session:

| `calibration.axis_check.verdict` | w_x | w_y | w_z |
|---|---|---|---|
| `w3c` | `rr_alpha` | `rr_beta` | `rr_gamma` |
| `w3c_inverted` | -`rr_alpha` | -`rr_beta` | -`rr_gamma` |
| `legacy` | `rr_beta` | `rr_gamma` | `rr_alpha` |
| `legacy_inverted` | -`rr_beta` | -`rr_gamma` | -`rr_alpha` |
| `unclear`, null, absent (schema 1.0) | as `w3c`, flagged `axis_unverified` | | |

`units == 'rad_per_s'` (with a definite verdict) converts to deg/s. The calibration gyro bias is mapped
the same way and subtracted before integrating, but only when the flat window was still
(`calibration.flat.still`) and the bias is below 3 deg/s; otherwise the mean rate of the 400 ms before GO
is used, if below 3 deg/s (`conventions.calibration_gyro_bias`, `GYRO_BIAS_MAX_DPS`).

**Accelerometer sign.** WebKit (every iOS browser) reports `acceleration` AND
`accelerationIncludingGravity` negated on all axes (face-up `accg_z` ~ -9.81 instead of +9.81); the
rotation rate is not inverted. Both acceleration fields (and the baseline gravity) are multiplied by the
session sign `s`:

1. `calibration.gravity_sign` (+1 / -1), source `calibration` - flagged if `upright_check` is `inverted`
   or if it differs from the engine expectation. If >= 5 trial votes contradict it by >= 80 %, the votes
   win (a flat step done face-down passes every calibration check with the wrong sign): source
   `inferred`, flagged, and `gravity_sign_detail` starts with "calibration sign overridden by the trial
   votes";
2. else the trial vote: the raw baseline gravity of each fixed-orientation trial against the
   instructed "up" direction (>= 3 votes, >= 80 % agreement), source `inferred` - flagged, and a
   disagreement with the engine expectation is named;
3. else the engine expectation `device.engine`: `webkit` -1, `blink` / `gecko` +1 (schema 1.0 sessions
   with `platform: ios` count as WebKit), source `engine` - flagged;
4. else +1, source `default` - flagged, and `modeling_table` drops these sessions from chiral modelling.

This is the order of docs/03 section 5.2 and docs/05 section 2.1 (measured evidence before the engine
prior), and the default. `load_dataset(..., gravity_sign_order=('engine', 'vote'))` puts the engine first;
`infer_gravity_sign=False` never lets the votes set the sign (step 2 is skipped, and a contradicted
calibration is kept and flagged). `qa_report.py` warns about every overridden calibration.

A wrong sign is a parity flip: it negates every polar vector but not w, so every chiral feature changes
sign and a right-hand trial looks like a left-hand one. Check `ds.sessions.gravity_sign_source`.

**Mirror (left <-> right, any grip).** acc / accg `(x, y, z) -> (-x, y, z)`; w `(x, y, z) -> (x, -y, -z)`,
i.e. rotationRate `{alpha kept, beta and gamma negated}` under the W3C mapping; hand flipped;
`top_left_*` <-> `top_right_*`. `augment.mirror_trial(record, rr_verdict=...)` mirrors raw records;
`prepare_trial(..., mirror=True)` mirrors the normalised signals (exact for any mapping).

**Gravity during the movement (`gravity_v1`).** The specific force of the rest windows before the onset
and after the end (|w| < `still_gyro_dps`, 300 ms, averaged in one gyro-integrated frame) is propagated
with the gyroscope (dg/dt = -w x g) forward from the onset and backward from the end, the two estimates
are blended linearly in time and rescaled to the calibration |g|. Linear acceleration is a = f - g. The
closure angle between the two rest estimates is a QA number (> 3 deg: suspect the gyro bias, the axis
mapping or a rest that was not still). The browser's own gravity (`accg - acc`) is only compared with it
(`qc_gravity_vs_browser_deg`), because accelerometer-corrected fusion mis-estimates tilt while the hand
accelerates.

**Platform harmonisation.** Chrome rounds to 0.1 m/s^2 and 0.1 deg/s, WebKit does not.
`augment.quantize_trial` applies Chromium's rounding (`train_baseline.py --quantize ios` for the
iOS-vs-Android ablation). Isolated exact repeats of a sample (duplicated WebKit snapshots; on Chrome only
while |w| > 20 deg/s) are dropped before resampling.

## online_v1: the sync rule

`segment.py` is an exact port of the spec in the doc comment of `app/src/core/segmentation.ts`; read that
comment before changing either side. To reproduce a stored trial, `segment.rerun_online_v1(record)`:

* uses the trial's own resolved `segmentation.params` (per-speed / per-variant overrides included);
* pushes every stored motion sample, in stored order and with the stored (rounded) values: position ==
  stream index, so `onset_index` / `end_index` refer to the stored arrays;
* gives GO right after pushing the **first sample with `t >= timing.go_ms`**, with **that sample's t**
  (the app emits GO at a sample, so this is exactly the GO sample), and only if the detector is in
  `still` there.

`segment.compare_with_stored` / `port_agreement` check it; `qa_report.py` reports the agreement. On the
reference runs of the app's TypeScript code in `tests/fixtures/ts_online_v1_reference.json.gz` events,
GO times and indices are identical and floating values agree to ~1e-15 (the last bits of `Math.exp`
differ between JavaScript engines and the C library). Regenerate the fixture after any change to
segmentation.ts, trialController.ts or the simulator: `cd app && npx tsx ../analysis/tests/fixtures/make_ts_reference.mts`.

`offline_v1` (the analysis segmentation, `segment_bounds_ms(record, 'auto')` falls back to the stored
online_v1) resamples to 100 Hz, low-passes the vector components at 8 Hz (zero phase), and on |w| and the
gravity-free |a| uses per-channel thresholds of max(% of the movement peak, k x the robust SD of the
pre-GO baseline): onset = start of the supra-threshold run leading to the first significant sample, end =
first sustained (200 ms) quiet period after the last significant sample, searched before the app's end
cue (+150 ms) so the arm's return is never merged in.

## Features

`features.feature_groups()` lists them (`extract_features` returns one row per trial plus `qc_*`
columns): **basic** (duration, reaction time, peaks, time to peak, total rotation, movement units),
**device** (per-axis statistics, 'up' at onset/end, net rotation vector: grip-dependent), **invariant**
(magnitudes and gravity-referenced components: unchanged by any fixed grip rotation and by mirroring) and
**chiral** pseudoscalars (net yaw about gravity and its time course, w·a_hat, helicity, acceleration
turning, path curvature: unchanged by grip rotations, sign-flipped by mirroring). Velocity-based features
need ZUPT integration and are NaN for movements longer than 1.5 s. `sequence_tensor` gives
time-normalised channels for sequence models.

## Adding a model

Any scikit-learn-style classifier that accepts the feature DataFrame works; pick columns with
`ColumnSelector` so new features never break it:

```python
# my_models.py (anywhere on PYTHONPATH, e.g. analysis/)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from motionlab.models import ColumnSelector

def make_forest():
    return Pipeline([('select', ColumnSelector(groups=('invariant', 'chiral'))),
                     ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0))])
```

```bash
python scripts/train_baseline.py data/synthetic --models yaw_rule,logreg,my_models:make_forest
```

or register it for the Python API: `motionlab.models.MODEL_REGISTRY['forest'] = make_forest`. Labels are
y = 1 for the right hand; `predict_proba[:, 1]` is p(right). Fit everything (scalers, calibration,
thresholds) inside the model so it only ever sees training participants. Report the mirror consistency:
a model that has learned chirality flips its decision on mirrored trials.

## Tests

```bash
python -m pytest tests -q            # from analysis/;  from the repo root: python -m pytest analysis/tests -q
```

They cover: schema validity of synthetic and app-generated records; loading both layouts plus bundles,
zips, envelopes and the deduplication rules; norm_v1 (W3C mapping, legacy verdict, gravity-sign
resolution including the override of a face-down calibration, the calibration gyro-bias checks, iOS =
Android after normalisation, drift of the copied TypeScript constants); formula-safe CSV output; the mirror
rule on random rigid-body trajectories, mirror involution and mirror(right) = left on noise-free
synthetic trials; online_v1 parity with the app's TypeScript runs, known-truth streams and the spec's
literal readings; offline_v1 against synthetic truth; chiral features flipping under mirroring and every
grip-invariant feature unchanged under a random grip rotation; gyro gravity propagation; subject-disjoint
splits for every scheme; and the three scripts end to end.

## Known limitations

* `offline_v1` reports `no_end` for some slow trials whose motion has not settled before the end cue
  (about 0.9 % of the synthetic standard set: 6 of 672 accepted trials, all slow); those fall back to the
  stored online_v1 segment.
* Euler angles, compass and absolute heading are metadata only (gimbal lock in the upright pose).
* `synth.py` is a simple independent generator for tests; the realistic synthetic data come from the
  app (`npm run synth`), which runs the real recording pipeline.
