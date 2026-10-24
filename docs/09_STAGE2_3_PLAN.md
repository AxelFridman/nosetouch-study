# 09 Stage 2 and 3 plan: training, the predictor link, the report and the deck

This document is the consolidated plan for everything after data collection. It covers how the street recordings become a calibrated left/right model (Stage 2) and how that model becomes a link the PI opens on their own phone, which answers with a probability (Stage 3). It also covers the LaTeX report and the PPTX deck. It was written on 2026-10-24 from three design drafts: training, the inference link, and the report and deck. Each draft was checked against the code, the street protocol (protocol 2.0.0, schema 1.2.0, app 0.2.0) and the one pilot session. [05](05_MODELING_PLAN.md) and [08](08_STAGE3_PREDICTOR.md) stay the reference documents. Where this plan changes them, [section 11](#11-proposed-edits-to-docs05-and-docs08) lists the exact edits; until those edits are applied, this document wins on the points it decides.

Related: [Modeling plan](05_MODELING_PLAN.md) · [Stage 3 predictor](08_STAGE3_PREDICTOR.md) · [Protocol](01_PROTOCOL.md) · [Sensors and platforms](03_SENSORS_AND_PLATFORMS.md) · [Segmentation](04_SEGMENTATION.md) · report skeleton [report/](../report/README.md) · deck outline [presentation/OUTLINE.md](../presentation/OUTLINE.md).

**Legend.** **D*n*** is a decision of this plan; it binds until it is changed here. **Rec.** = recommendation and **Hyp.** = hypothesis to test on data, as in 05. "Pilot" means the single real session in `analysis/data/real/`: P-VPSDL4, Android Chrome, protocol 1.0.0 with the table calibration, 23 attempts, 18 accepted, and 16 after the gap rule T6. One person says nothing about accuracy, so the pilot is used only to check that the pipeline and the physics behave.

---

## 1. Summary

- **Output.** For one movement by a person the model has never seen, the predictor gives P(right), calibrated for new people, or "not sure" with a reason. A forced-choice answer is always available.
- **Model.** The default candidate is an antisymmetric logistic regression on chiral features: grip-invariant pseudoscalars that change sign under the mirror. It works on 60 Hz data, is exported as JSON and runs in TypeScript. Tree and sequence models compete, but ship only if they are clearly better (D12, D15).
- **Evaluation.** Every split is by person, never by trial. A locked 25 % test set is chosen by a rule fixed in this document. Calibration and thresholds are fitted only on training people (D3, D16–D20).
- **Link.** It is the same PWA at `?predict=1`. It works offline and makes no network requests at all. A 2.5-s upright check replaces the table calibration (D22–D28).
- **Report and deck.** The report is a KOMA-Script article of at most 20 pages plus appendices, with every number generated ([report/](../report/README.md)). The deck has 14 slides plus backups and is built from [presentation/OUTLINE.md](../presentation/OUTLINE.md) (D30–D33).
- **Two pilot checks changed the plan** (§2):
  - The hand shows in the *time course* of the yaw about gravity, not in its net value.
  - online_v1 with the predict-mode parameters of 08 §2.4 cannot re-segment training trials. The deployed model therefore uses offline_v1, ported to TypeScript.

## 2. Pilot checks behind the decisions

Both checks use motionlab on `analysis/data/real` and can be repeated with `extract_features` and `segment.rerun_online_v1(record, params=...)`.

### 2.1 The hand is in the yaw time course, not in the net yaw

16 gap-free accepted trials, 8 per hand, one right-handed person. The AUC is P(feature of a right-hand trial > feature of a left-hand trial), and the CI comes from a trial bootstrap, which ignores that all trials share one person.

| Feature (chiral, grip-invariant) | AUC (95 % CI) | Sign alone | Median right | Median left |
|---|---|---|---|---|
| `chi_yaw_first_half_deg`: ∫ω·ĝ over the first half of the movement | 0.95 (0.81–1.00) | 15/16 | +5.2° | −2.2° |
| `chi_yaw_rate_b2`: mean ω·ĝ in the 2nd of 5 time bins | 0.84 (0.62–1.00) | 13/16 | +6.9 deg/s | −2.6 deg/s |
| `chi_yaw_net_deg`: Ψ = ∫ω·ĝ over the whole movement | 0.58 (0.23–0.90) | 8/16 | +2.4° | +0.7° |
| `chi_yaw_second_half_deg` | 0.25 (0.00–0.58) | – | −2.4° | +3.8° |
| `chi_yaw_peak_dps`: the signed peak of ω·ĝ | 0.14 (0.00–0.38) | – | −65 deg/s | +76 deg/s |

**Reading.**
- Early in the reach, the phone turns counter-clockwise seen from above for the right hand and clockwise for the left: the forearm swings toward the midline as the elbow flexes.
- The wrist then turns the phone back so that the screen faces the person again. Start and end both face the person, so the net yaw cancels.
- 05 §4.3 bet on the sign of the net yaw. The signal is in the time course instead, and the largest yaw rate belongs to the turn back, with the opposite sign. Any rule must therefore learn its sign from data, which `YawSignRule` already does.
- **Hyp.** The first 10 development people confirm this (M2). The design draft reported 18 trials, AUC 0.96 and 17/18. The numbers above apply rule T6 (no gaps over 50 ms), which leaves 16 trials.

### 2.2 The predict-mode parameters of 08 §2.4 cannot re-segment training data

All 18 accepted pilot trials were re-run through the exact Python port of online_v1.

| Segmentation | Trials with an end | Comment |
|---|---|---|
| online_v1, the parameters stored with each trial (per speed) | 18/18 | Identical to the app. |
| online_v1, protocol 2.0.0 natural parameters for every trial | 18/18 | 3 of 6 slow trials end after 0.3–0.9 s; offline_v1 finds 4.1–5.1 s for them. |
| online_v1, slow end thresholds (10 deg/s and 0.25 m/s², held 450 ms) | 8/18 | Natural and fast trials never get that still before the recording stops. |
| online_v1, the 08 §2.4 predict parameters under 2.0.0 (slow + pause: held 1800 ms) | 0/18 | Nobody holds still at the nose for 1.8 s in the collector. |
| offline_v1 (peak-relative thresholds, end searched before the end cue) | 18/18 | Fast 1.0–2.3 s, natural 2.4–3.1 s, slow 3.6–5.4 s. |

The pilot used protocol 1.0.0, which records 1000 ms after the end. Street sessions record 500 ms and then the return movement, so the hold at the nose is even shorter and the conclusion holds a fortiori. No single online_v1 parameter set both completes on the training data and keeps slow movements whole. The deployed model therefore needs an offline, peak-relative segmentation that runs identically in Python and in the browser (D10).

## 3. Conflicts between the drafts and how they were resolved

| # | Conflict or gap | Resolution |
|---|---|---|
| 1 | Predict URL: `?mode=predict` (inference draft) vs `?predict=1` (08, report draft) | D22 |
| 2 | Model fetched at runtime via `app/public/models/index.json` (08) vs bundled with the code (inference draft) | D24: bundled. A runtime fetch is impossible under the no-network policy. |
| 3 | Check and engine disagree on the gravity sign: abstain (training draft) vs block unless "continue anyway" (inference draft) | D23: repeat once, then refuse. There is no "continue anyway". |
| 4 | "Approach and return disagree → medium confidence" (training draft), but predict mode does not record the return (inference draft) | D11: v1 uses the approach only; ablation A10 studies the return. |
| 5 | Out-of-distribution (OOD) detection: Mahalanobis distance (training draft) vs per-feature ranges (inference draft, 08 §4) | D20: ranges on mirror-even descriptors; Mahalanobis is ablation A12. |
| 6 | Symmetric isotonic calibration offered (training draft) vs no isotonic (inference draft) | D19: isotonic only as ablation A11. |
| 7 | Ablation IDs A9 and A10 mean different things in two drafts | One table, §4.7. |
| 8 | New exclusion rules "T8–T10" collide with the existing T8–T12 of [02 §12](02_DATA_SCHEMA.md#12-data-quality-and-exclusion-rules) | D5: no new exclusions. Movement checks become predictor gates G1–G4, applied the same way in evaluation. |
| 9 | Locked test: 20 % by `participant_id` (05 §6.4) vs 25 % by person with a salt the PI keeps (training draft) | D3: 25 % by person, no salt. |
| 10 | PI session: 10 attempts per person (08 §3.1) vs 12 (inference draft) | D29: 12. |
| 11 | Gyro bias from the hand-held check window (inference draft) vs the pre-GO window (training draft) | D9: pre-GO in both, for parity. |
| 12 | Predict mode records 300 ms after the end, but sequence windows reach end + 0.5 s | No conflict: online_v1 confirms the end only after its sustain period (1.8 s in predict mode), so that hold is already recorded. |
| 13 | Report draft: "published sign errors (OREBA, Kyritsis, OpenWatch)" | Fixed in the report skeleton. Per 05 §3.2, OREBA and Kyritsis use the correct transform; only OpenWatch's violates the pseudovector rule. |
| 14 | Report draft names (`nosetouch_report.tex`, `generated/`, `reach_left.gif`) vs the repository | The repository's names: `report/main.tex`, `report/figures/`, `report/tables/`, `docs/img/gesture/*.gif`. |
| 15 | `pull_live.py` was said to write time-stamped folders with a snapshot hash; the script replaces `analysis/data/live/` and has no hash | D1: a separate freeze step. |
| 16 | Choice trials are "kept out of training" in two drafts, but `motionlab.io` does not expose `condition.hand_source`. Today they would enter training and the primary metric with a self-reported label. | D4, milestone M1. |
| 17 | Hands-free predict series and aggregation over several attempts (inference draft) vs the single-trial use case | D27: not in v1. |
| 18 | The draft PI protocol fixed the conditions per hand, which takes the choice away from the PI | D29: the list is only a suggestion; the PI chooses. |

## 4. Stage 2: training and evaluation

This section adds to [05](05_MODELING_PLAN.md) and replaces none of it unless §11 says so.

### 4.1 Data

**D1 Data sources and freezing.**
- **Live data.** `python analysis/scripts/pull_live.py --real-only` downloads the Heroku export into `analysis/data/live/`, replacing the folder on every pull.
- **Freeze** (new, M1). Copy a pull to `analysis/data/frozen/<UTC stamp>/` and write `snapshot.json`: the sha256 over the sorted list of (record id, `updated_at`, sha256 of the record's canonical JSON), plus counts. Every run manifest, the report and the model file carry this hash.
- **The pilot** is used for development only, never for the locked test.
- **Synthetic data** serve only for pipeline tests and parity fixtures. Every reported number uses `simulated='exclude'`.
- **The stress set** is the set of PI-style rehearsal sessions recorded with predict mode's local log (D29). It is never trained on.

**D2 The person is the unit of analysis.**
- **Person key.** `person_key` is the earliest P-code of a person. A friend who redoes the session on another phone, or whose Safari storage was cleared, gets a new P-code.
- **Aliases.** A git-ignored alias table, `analysis/data/person_aliases.csv` (columns `alias, person_key`), merges such codes. The student keeps it. QA lists candidate duplicates: the same age, sex, dominant hand and phone model, or the same uploaded name. Names never leave the database and the alias table.
- **Grouping.** Every split, bootstrap and per-person metric groups by `person_key`.
- **Cap.** At most 40 trials per person enter a training fold (seeded random choice), so the student's own sessions cannot dominate.

**D3 The locked test set.**
- **Rule** (fixed by the commit that adds this document): a person is in the locked test iff `int(sha256(("nosetouch-locked-v1:" + person_key).encode()).hexdigest(), 16) % 4 == 0`, i.e. about 25 %.
  - The app draws the P-code at random at enrolment, so nobody chooses the assignment, and the commit date shows that the rule predates the data.
  - A salt held by the PI would add nothing: the student must know the assignment to keep those people out of development.
- **Always excluded from the test set,** whatever the hash says: the student (every session) and the pilot participant.
- **Access.**
  - Development code loads only development people by default: `load_dataset(..., partition='dev')`, part of M1.
  - The locked-test run is a separate script. It writes `locked_test_opened_at` into its manifest and refuses a second run for the same model sha256.
- **No redraw.** If the test set comes out small or unbalanced, the report states its composition and CI width as they are.

### 4.2 Labels and exclusions

**D4 Labels.**
- **Label.** The label is `condition.hand` when `condition.hand_source` is `instructed`; an absent value counts as instructed.
- **Choice trials** (`hand_source = reported`, the tier-0 trial) form the "natural use" set.
  - They are never trained on and are not part of the primary metric.
  - They are scored out of fold and reported separately, as an accuracy with a CI and a comparison with the profile's `phone_hand`.
  - Each one is a person's first, unpractised movement with a hand they chose: the closest thing in the data to the PI's first attempt.
- **Flagged trials** (rule T3) stay excluded.

**D5 Exclusions vs gates.**
- The recording rules T1–T7 of 02 §12 stay as they are. Such attempts are excluded, and the app has the participant redo them.
- **There are no new exclusion rules.** Checks of movement plausibility become **predictor gates** (§5.6), and evaluation applies them exactly as the predictor will:

  | Gate | Checks | Predict mode | Evaluation |
  |---|---|---|---|
  | G1 recording | rate ≥ 50 Hz; no gap > 50 ms in [onset − 0.3 s, end + 0.3 s]; page not hidden; gyroscope present | retry | excluded |
  | G2 segment | offline_v1 finds a segment; 0.25 s ≤ duration ≤ 8 s; peak \|ω\| ≥ 30 deg/s | retry | excluded |
  | G3 unusual movement | OOD rule of D20 | abstain | scored |
  | G4 low confidence | below τ (D20) | abstain | scored |

  "Scored" means that the forced-choice metric uses the more likely hand, while the selective metrics (coverage, accuracy on answered trials) count the attempt as not answered.
- **Check.** Report retry and abstention rates by hand, which should be similar. A gate that fires more for one hand would bias the score.

**D6 Screening for the wrong hand** (with hands-free review, `accepted` only means "completed and not redone").
- **Where.** Inside each outer training fold, an antisymmetric logistic model is fitted with inner leave-one-person-out.
- **What it flags:**
  - training trials it gives p(label) < 0.05;
  - runs of 3 or more consecutive trials it confidently reverses, which suggest a misread hand cue.
- **Effect.** Flags remove trials from training only, never from evaluation, so the cleaning cannot inflate accuracy. A sensitivity analysis reports results with and without the screening.
- **Review.** `qa.py` lists the flagged trials so they can be looked at with `plot_trial.py`.

### 4.3 Preprocessing and segmentation

**D7 Gravity sign in the training data.**
- **Resolution.** norm_v1 as implemented ([03 §5.2](03_SENSORS_AND_PLATFORMS.md#52-gravity-sign-resolution)):
  1. the calibrated sign, which for street sessions is `implicit_v1`, unless 5 or more trial votes contradict it;
  2. the trial votes (upright and sideways grips vote);
  3. the engine prior;
  4. the `default` sign, whose sessions are excluded.
- **New table, per session:** agreement between `implicit_v1` alone, the votes and the engine prior. Predict mode has only a check and the engine (D23), so this table estimates how often that pair could be wrong.

**D8 rotationRate axes.**
- **Assumption.** The W3C mapping, which every current engine follows.
- **New QA,** `conventions.implicit_axis_check`:
  - It computes the session's median gravity closure under each candidate mapping.
  - It flags the session when another mapping beats W3C by at least 1.5× and 2°.
  - It never remaps automatically.
  - The design draft measured 2.7° for W3C on the pilot vs 5.7–7.8° for the alternatives. Re-check this once the function is implemented.
- **Limit.** The check cannot see the accelerometer sign ([05 §3.3](05_MODELING_PLAN.md#33-what-the-ios-sign-inversion-means-parity)).

**D9 Gyro bias.** Subtract the mean rate of the 400 ms before GO when its norm is below 3 deg/s, else subtract nothing. Training (`prepare_trial(remove_gyro_bias='pre_go')`) and the predictor do the same. Street sessions have no table window, and a hand-held check is not rest.

**D10 The deployed model's segment comes from offline_v1** on the recorded window, in Python and in TypeScript.
- **Port.** Python is the reference. The TypeScript port uses fixed filter coefficients (the 100 Hz grid and cut-offs are fixed) and scipy's `filtfilt` padding, and golden fixtures pin it.
- **online_v1 in predict mode** only drives the cues and stops the recording. Its parameters (§5.3) make the person hold the phone at the nose for about 2 s, so the recording always contains the whole movement.
- **Fallback.** If an exact port proves impractical, a deploy variant with moving-average filters (`offline_d1`) replaces offline_v1 in both languages, and ablation A5 reports the cost.
- **Evidence:** §2.2.

**D11 The deployed pipeline, `feat_d1`.**
1. Steps: norm_v1 → drop repeated samples → linear resampling to 60 Hz → pre-GO bias → gravity_v1 → offline_v1 bounds → features over the approach window only.
2. What it leaves out:
   - zero-phase filtering inside the features;
   - ZUPT velocity, which needs movements of 1.5 s or less, while pilot movements lasted 1–5 s;
   - Euler angles, compass and absolute heading.
3. Model inputs:
   - the chiral yaw family: first half, second half, the 5 time bins and the signed peak;
   - ω·â, helicity and the normalised triple-acceleration feature;
   - odd × even products with log duration and peak |ω|. These stay antisymmetric.
4. Mirror-even descriptors are used only by the gates.
5. The headline numbers, τ, the reliability table and the expectation for the PI all come from this pipeline. The research features (100 Hz, filtered, with ZUPT) are context and ablation A5.
6. Windows:
   - W0, the pre-GO rest: gyro bias and the starting ĝ;
   - W1, the approach: onset → end;
   - W2, the hold at the nose: g_post;
   - W3, the return: recorded by the collector only.

   Sequences cover [onset − 0.5 s, end + 0.5 s].
7. **Time reversal** (ω → −ω, a unchanged) also flips every pseudoscalar. A return that retraces the approach therefore looks like the other hand. Never pool W1 and W3; A10 tests a time-reversed or separately modelled return.

### 4.4 Models

**D12 The model ladder** (extends [05 §5](05_MODELING_PLAN.md#5-baselines-ladder)).

| Rung | Model | Code (✓ exists, + new) | Ships as |
|---|---|---|---|
| 0 | Chance; labels shuffled within person | ✓ `make_chance`, `shuffle_labels_within` | – |
| 1 | `YawSignRule` on `chi_yaw_first_half_deg`; sign learnt from training data | ✓ `YawSignRule(feature=...)` | 3 numbers |
| 2 | **Default candidate.** L2 logistic regression, no intercept, chiral features plus odd × even products (antisymmetric) | ✓ `make_logreg_chiral`, + `make_logreg_chiral_ix` | JSON |
| 2b | Unconstrained logistic regression (adds even and device-frame features), mirror-augmented | ✓ `make_logreg` | JSON |
| 3 | HGB: unconstrained + augmented vs `MirrorSymmetrized(HGB)` | ✓ `make_hgb`, + `models.MirrorSymmetrized` | JSON trees |
| 4 | MiniRocket on the 8 invariant channels + ridge, symmetrised | + `models.make_minirocket` (aeon) | research only |
| 5 | Siamese 1D-CNN or GRU, ≤ 50k parameters | + `motionlab/deep.py` (PyTorch) | research; ported to TS only if it wins |

**Antisymmetry by construction.** The logit is z(x) = g(x) − g(m(x)) ([05 §3.5](05_MODELING_PLAN.md#35-uses)).
- **Wrapper.** `MirrorSymmetrized(base)` builds it for any scikit-learn model from a closed-form `features.feature_mirror(F)`:
  - `chi_*` change sign;
  - for `dev_acc_x_*`, `dev_gyro_y_*` and `dev_gyro_z_*`, the mean changes sign, the SD stays, and (min, max) becomes (−max, −min);
  - `dev_up_*_x`, `dev_rot_y_deg` and `dev_rot_z_deg` change sign;
  - basic and invariant features stay the same.
- **Test.** A unit test checks that `feature_mirror(F)` equals `extract_features(..., mirror=True)`.
- **Augmentation is redundant** for antisymmetric models: the loss on (x, y) equals the loss on (m(x), 1 − y).

**D13 Augmentation lives inside the estimator.**
- **Wrappers.** `MirrorAugmented` adds the mirror image with the flipped label. The others are grip tilt up to 15–20°, time warp × 0.8–1.25, magnitude × 0.9–1.1, and the screen-away flip Q = diag(−1, 1, −1), i.e. 180° about y, for device-frame and sequence models.
- **Why inside.** `fit()` augments only the rows it is given, so every outer or inner CV validates on real trials only.
- **Not like 05 §6.1.** The sketch in 05 §6.1 augments first and then runs `GridSearchCV` with groups, which puts mirrored copies into the inner validation folds; §11 replaces it.
- **Test.** Q is a proper rotation, so the chiral features must not change under the screen-away flip.

**D14 Research rungs.** Rungs 4–5 start at 15 or more development people. They ship only if they beat rung 2 by more than the CI width and pass the TypeScript parity test. Only then does onnxruntime-web, with its 14.2 MB WASM, come into play.

**D15 Model selection.**
- **Rule.** On the development CV with the deploy pipeline, ship the simplest rung whose accuracy lies inside the 95 % CI of the best rung (paired person bootstrap).
- **Freeze.** The choice is frozen before the locked test, and exactly one model is evaluated there.

### 4.5 Evaluation

**D16 Schemes.**
- `stratified_group_kfold`: 5 folds × 5 seeds, groups `person_key`, strata `dominant_hand × platform`. The street profile has no Edinburgh score.
- `loso`, `orientation` (mirror pairs held out together) and `platform`.
- New `speed` and `variant` leave-one-level-out schemes, because the PI's speed and movement are unknown.
- New `evaluate.learning_curve`: train on 5, 10, 15, … people and test on the rest.
  - A plateau means that more people only narrow the CI.
  - A rising curve means: collect more.
- **Nested tuning.** `GroupKFold` on inner people, scored by log loss.

**D17 Metrics and CIs.**
- **Metrics:**
  - accuracy (primary), balanced accuracy, AUC, log loss, Brier score;
  - ECE with 10 equal-mass bins;
  - per-person accuracy: median, 10th percentile, and the number of people below 80 %.
- **CIs:**
  - person bootstrap with B ≥ 2000, as implemented (one level, resampling people);
  - paired bootstrap for differences between models or ablations, with both arms on the same resampled people;
  - never fold standard errors alone.

**D18 Controls.**
1. Labels shuffled within person give ≈ 50 %.
2. Record-wise vs subject-wise gap.
3. Platform probe.
4. Mirror consistency: at most 1e-9 for antisymmetric models, asserted in tests.
5. The sign-source agreement table (D7).
6. Retry and abstention rates by hand (D5).
7. Accuracy on the choice trials.

Nested cross-validation, as it must run:

```python
# sketch; names marked + are new
for outer_train, outer_test in make_splits(scheme, person_key, ...):          # people never shared
    fold = cap_per_person(outer_train, 40)                                        # D2
    fold = drop_wrong_hand_flags(fold, inner='loso')                              # D6, training rows only
    est = Calibrated(MirrorAugmented(base) if not base.antisymmetric else base,  # D13, D19 (+)
                     method='temperature' if base.antisymmetric else 'platt',
                     inner_cv=GroupKFold(4))         # inner cross-fitted logits -> T, then refit on the fold
    search = GridSearchCV(est, grid, cv=GroupKFold(4), scoring='neg_log_loss')
    search.fit(X[fold], y[fold], groups=person_key[fold])   # inner validation folds hold real trials only
    p_right[outer_test] = search.predict_proba(X[outer_test])[:, 1]               # real trials only
# tau, OOD ranges and the reliability table come from these OOF predictions (D20); the locked test is untouched
```

### 4.6 Probability output

**D19 Calibration.**
- **Inside each outer training fold:**
  1. compute inner cross-fitted logits (GroupKFold by person);
  2. fit a temperature T (antisymmetric models) or Platt's (a, b) (unconstrained models) on them;
  3. refit the base model on the whole fold.

  The outer test fold is never seen. The shipped model goes through the same steps on all development people.
- **Why temperature.** It has no bias term, so p(m(x)) = 1 − p(x) survives.
- **Isotonic regression,** including its symmetric form, is ablation A11 only.
- **Never** fit calibration on in-sample predictions or on test people.

**D20 Abstention and τ.**
- **Gates** G1–G4 (D5).
- **OOD rule.** Use the 0.5–99.5 % training ranges of 6 mirror-even descriptors: duration, peak |ω|, peak |a|, total rotation, tilt change and gravity closure. Abstain when 2 or more are outside, or when the closure exceeds its 99.5th percentile (at least 5°).
  - Mirror-even inputs make abstention independent of the hand.
  - A Mahalanobis rule is ablation A12, compared on the risk–coverage curve.
- **τ** is the smallest threshold at which the development out-of-fold (OOF) accuracy on answered trials reaches 95 % (**Rec.**; question b for the PI).
  - Accuracy at τ measured on those same OOF predictions is optimistic, because τ was tuned there, and is labelled so.
  - The honest numbers for τ come from the locked test and the PI session.

**D21 What p means.** p is the probability that this movement was made with the right hand:
- for a new person from the training population;
- with left and right equally likely beforehand;
- given that the attempt passed the gates.

If the prior share π of right-hand attempts is known, add ln(π/(1 − π)) to the logit. The PI protocol balances the hands, so π = 0.5.

### 4.7 Ablations (one table; A1–A8 as in [05 §8.2](05_MODELING_PLAN.md#82-ablations))

| ID | Comparison | Decision it informs |
|---|---|---|
| A1 | Mirror augmentation on/off; antisymmetric vs unconstrained; overall and for left-handers | Ship an antisymmetric model? |
| A2 | 60 Hz vs high rate (phyphox) | Out of scope unless a phyphox cohort is run |
| A3 | iOS quantised vs raw | norm_v1 default |
| A4 | Feature sets: chiral / + invariant / + device-frame / sequences / + touch, leave-one-orientation-out | Robustness to new grips |
| A5 | Segmentation and features: research (offline_v1, 100 Hz, filtered, ZUPT) vs deploy (`feat_d1`) vs online_v1 with stored parameters | The cost of deployability |
| A6 | Gyro only / acc only / both | Robustness to sign and quantisation problems |
| A7 | Approach only vs approach + hold | Value of the end pose |
| A8 | Train on right-handers, test on left-handers | Size of the dominance shortcut |
| A9 | Screen-away flip augmentation (device-frame and sequence models); check on the stress set | Robustness to screen-away grips |
| A10 | Approach only vs approach + time-reversed return (or a separate return head) | Whether the return adds evidence |
| A11 | Calibration: none / temperature / Platt / (symmetric) isotonic | Calibration method |
| A12 | OOD: per-feature ranges vs Ledoit–Wolf Mahalanobis on even features (risk–coverage) | Abstention rule |
| A13 | Heading-canonical body frame vs the gravity frame | Kept only as a check; the design draft found no gain on the pilot |
| A14 | Rung-1 feature: first-half yaw vs net yaw vs signed peak yaw rate | Confirms §2.1 on more people |

### 4.8 What each analysis run must write for the report

The report and the deck are built only from `analysis/reports/<run>/` folders. Contract (owner: the analysis workstream):

- `manifest.json`:
  - `run_id`, `created_at`, `git_commit`, `git_dirty`;
  - `data_snapshot_sha256`, `simulated_fraction`;
  - schema, protocol and app version ranges;
  - `pipeline_versions`: norm, gravity, segmentation, features;
  - the arguments, `n_boot`, seeds, and the python and package versions;
  - `locked_test_opened_at`, for the locked run only.
- **Existing files, with additions:**
  - `metrics.csv` gains `ece`, `coverage_at_tau` and `n_answered`;
  - `oof_<scheme>.csv`, `participant_accuracy.csv` and `subgroups.csv` are unchanged;
  - `features.csv` gains an un-normalised Ψ column for figure F06.
- **New tidy files:**
  - `trials_meta.csv`: trial_id, person_key, hand, hand_source, orientation, speed, variant, start_position, tier, platform, engine, dominant_hand, phone_hand, sex, age_band, duration_ms, calibration_method, gravity_sign_source. No names.
  - `ablations.csv`: ablation, variant, metric, value, delta, delta_lo, delta_hi, n_people.
  - `yaw_timecourse.csv`, `sign_patterns.csv`, `tier_completion.csv`.
- **The locked test** gets its own run folder.

## 5. Stage 3: the predictor link

This section adds to [08](08_STAGE3_PREDICTOR.md).

### 5.1 Entry and isolation

**D22 Where the predictor lives.**
- **URL.** `https://axelfridman.github.io/nosetouch-study/?predict=1&model=<id>`. The evaluation QR code pins the model.
  - `?mode=predict` is an alias.
  - `&sim=1` is the silent desk demo, a known-answer check only.
  - `&lang=he` switches to Hebrew.
- **Same PWA, no separate build.** `main.tsx` branches before `boot()` and loads `./predict/mount` with a dynamic `import()`. Predict mode therefore makes:
  - no `config.json` fetch;
  - no use of the collector's IndexedDB;
  - no `uploader.start()`;
  - no growth of the collector bundle.
- **Storage.** A separate IndexedDB, `nosetouch-predict`, with `study_code: 'nose-touch-predict'` and a throwaway P-code.
- **Files.** `app/src/predict/` holds `PredictApp.tsx`, `predictor.ts`, `log.ts`, `pipeline/` and `models/`. The Python side is `motionlab/deploy.py` and `scripts/export_model.py`.

### 5.2 The device check (replaces the 15-s table check of 08 §1)

**D23** One tap enables the sensors, audio and wake lock. Then the person holds the phone upright, screen facing them, for 2.5 s.
- **Requirements.** Motion ≥ 50 Hz, a gyroscope, and a still window.
- **Sign.** The sign of the raw `accg_y` (with |accg_y| ≥ 7 m/s²) must equal the engine prior: WebKit −1, Blink and Gecko +1.
- **On disagreement:**
  1. repeat the check once, with a picture of the pose;
  2. if it still disagrees, refuse (`sign_conflict`). A wrong sign inverts every answer, so there is no "continue anyway".
- **Unknown engine.** Two consistent checks are required, and every result is flagged.
- **Axes.** The mapping is assumed to be W3C. Every attempt verifies it through the gravity closure (gate G3).
- **Bias.** The check does not provide the gyro bias (D9).

### 5.3 Pipeline, cues and parity

**Pipeline.** The predictor runs `feat_d1` (D11) on the recorded window.

**Cue parameters.** online_v1 in predict mode decides only when to play the GO and end cues and when to stop recording (**Rec.**; validate on the rehearsal):
- start from `DEFAULT_SEGMENTATION_PARAMS` of protocol 2.0.0;
- onset floors 25 deg/s and 0.55 m/s² (the slow values), so slow starts are caught;
- end thresholds 30 deg/s and 0.6 m/s² (the natural values);
- `end_sustain_ms` 1800, so pauses do not end the recording early;
- `max_movement_ms` 12000, `post_roll_ms` 300, `return_cap_ms` 0.

The model never sees these indices (D10).

**D24 Packaging and integrity.**
- **Format.** The model is a JSON file (at most 200 KB) bundled into the predict chunk with a `?raw` import. It is written minified, with sorted keys, no NaN and no trailing newline.
- **Hash.** `.gitattributes` marks these files `-text`, so a CRLF checkout keeps the hash. `app/src/predict/models/registry.ts`, written by `export_model.py`, maps each id to its sha256.
- **At start-up** the browser recomputes the sha256 with `crypto.subtle`. It refuses on a mismatch (`model_integrity`) or when the model's versions differ from the pipeline's (`version_mismatch`).
- **Rollback.** The current and the previous model are both bundled, and `&model=` selects one. An unknown id is refused.
- **Model kinds** the TypeScript evaluator supports: `yaw_rule`, `logistic`, and `hgb` (trees read from scikit-learn's private `_predictors`, so the scikit-learn version is pinned and a round-trip test runs).

**D25 Parity tests. Python is the reference.**
- **Golden file.** `scripts/export_model.py --golden` writes `app/tests/fixtures/predict_golden_<id>.json.gz` with about 70 trials:
  - real trials;
  - synthetic edge cases: iOS sign, legacy axis mapping, rad/s, missing acc, gaps;
  - 10 mirrored copies;
  - 6 trials that must abstain.
- **Contents per trial.** Segmentation indices, per-channel checksums after resampling, the features, the logit, p, the status and the reasons. A failure then shows which stage diverged.
- **Vitest tolerances:**
  - features and p within 1e-6 relative;
  - segmentation indices identical, with ±1 sample allowed on at most 1 % of trials;
  - status, reasons and hand identical.

### 5.4 The model file

**D26** A model file carries the fields below. The values are placeholders, not results. Its own sha256 lives in `registry.ts`, because a file cannot hold its own hash.

```json
{
  "format": "nosetouch-model/1",
  "model_id": "lr-chiral-d1-v1",
  "kind": "logistic",
  "antisymmetric": true,
  "versions": {"norm": "norm_v1", "gravity": "gravity_v1", "segmentation": "offline_v1",
               "features": "feat_d1", "model_format": "1"},
  "resample_hz": 60,
  "quantise_ios": false,
  "feature_names": ["chi_yaw_first_half_deg", "chi_yaw_second_half_deg", "chi_yaw_rate_b1"],
  "scale": [1.0, 1.0, 1.0],
  "coef": [0.0, 0.0, 0.0],
  "intercept": 0.0,
  "calibration": {"method": "temperature", "temperature": 1.0},
  "abstain": {"tau": 0.65, "target_accuracy": 0.95},
  "ood": {"features": ["duration_ms", "peak_gyro_dps", "peak_acc_ms2", "total_rotation_deg",
                       "inv_tilt_change_deg", "qc_closure_deg"],
          "lo": [0, 0, 0, 0, 0, 0], "hi": [0, 0, 0, 0, 0, 0], "max_outside": 1, "closure_max_deg": 5.0},
  "reliability": [{"p_lo": 0.5, "p_hi": 0.6, "n": 0, "k": 0}],
  "training": {"data_snapshot": "sha256:...", "git_commit": "...", "n_people": 0, "n_trials": 0,
               "platforms": {"ios": 0, "android": 0},
               "metrics": {"dev_cv": {"accuracy": [null, null, null], "auc": [null, null, null],
                                      "ece": null, "brier": null},
                           "locked_test": {"accuracy": [null, null, null]}},
               "created_at": "..."},
  "intended_use": "One nose-touch movement by an adult; phone held in any fixed grip.",
  "limitations": ["Screen-away grips only synthesised", "Trained mostly on standing participants"]
}
```

### 5.5 What the result screen shows

**D27 The result screen.**

- **Answer.** The hand in large type with the probability of that hand, max(p, 1 − p), e.g. "RIGHT 91 %". P(left) and P(right) appear in small type, shown as "<1 %" or ">99 %" at the extremes.
- **Colours.** LEFT #d1495b and RIGHT #00798c, as in every figure.
- **Confidence label.**
  - "high" at 0.90 or more;
  - "medium" from τ to 0.90;
  - "not sure" below τ, still naming the more likely hand.
- **Reliability.** One line taken from the model file's reliability table, e.g. "On new people, answers between 90 and 97 % were right 118 of 124 times".
- **Evidence.** The first-half yaw, the duration and the closure.
- **Footer.** Model id, the first 8 characters of its sha256, the app version and the pipeline versions.
- **Order of saving.** The prediction is saved, with a timestamp, before the "It was LEFT / RIGHT" buttons unlock.
- **Not in v1:**
  - a bootstrap range for p;
  - aggregation over several attempts;
  - a hands-free series that starts attempts by itself.

  Every attempt stands alone, and nothing is learnt during a session.

UI states:
1. `gate`
2. `permission`
3. `check`
4. `ready`, with an offline-ready badge
5. `still`
6. `moving`
7. `hold`: about 2 s, until the double beep
8. `computing`, with a 1-s watchdog
9. `result`, `not_sure` or `retry(reason)`
10. `history`

### 5.6 Gates and failure handling (changes to 08 §4)

| Situation | Detection | Behaviour (reason code) |
|---|---|---|
| In-app browser, insecure page, no motion API | `device.in_app_browser`, capabilities | Block: "open in Safari/Chrome", with a copy-link button |
| Permission denied, no events | `requestMotionPermission`; no event within 2.5 s | Block, with the `deniedContext` advice (`permission_denied`, `no_events`) |
| Check fails | rate < 50 Hz, not still, \|accg_y\| < 7 m/s² | Repeat the check (`check_failed`) |
| No gyroscope | `rotation_available` false during the check | Refuse (`no_gyro`) |
| Check and engine disagree twice | D23 | Refuse (`sign_conflict`) |
| Model hash or versions wrong | D24 | Refuse (`model_integrity`, `version_mismatch`) |
| G1 recording | rate, gap in window, hidden, gyro | Retry, not scored (`rate_below_50hz`, `gap`, `hidden`, `no_gyro`) |
| G2 segment | no onset, too long, no offline_v1 segment, duration or peak implausible | Retry, not scored (`no_movement`, `too_long`, `no_segment`) |
| G3 unusual movement | OOD rule (D20) | Abstain, with the more likely hand shown small (`unusual_movement`, `closure`) |
| G4 low confidence | max(p, 1 − p) < τ | "Not sure" (`low_confidence`) |
| Unconstrained model fails the mirror check | \|p(x) + p(m(x)) − 1\| > 0.2 | Confidence label at most "medium" |

Removed from 08 §4: "calibration missing or unclear → engine prior, low confidence" (now a refusal) and "multi-peak → medium" (v1 has no portable peak counter; the OOD rule covers odd movements).

### 5.7 Network and privacy

**D28 Predict mode makes no network requests.**
- **CSP.** Before the sensors start, `mount()` inserts `<meta http-equiv="Content-Security-Policy" content="connect-src 'none'">`, after the predict chunk has loaded, so the CSP does not block Vite's preload. A page can add CSP policies but never remove them, so fetch, XHR, WebSocket and `sendBeacon` stay blocked for the life of the page.
- **Defence in depth.** The predict chunk does not even import the uploader. The Playwright WebKit test and a real iPhone must confirm that WebKit enforces a script-inserted CSP.
- **Local log.** It keeps each raw `TrialRecord`, the prediction and the truth, and exports CSV and JSON through `core/exporter.ts`.
- **Central logging** (`&log=1`) is deferred. It needs consent text, and it would have to choose its CSP before the first one is inserted.

Scoring-log CSV columns (the report reads them for T14 and F22): `ts, model_id, model_sha8, app_version, platform, browser, phone_model, person_code, slot, true_hand, predicted_hand, p_right, confidence, status, reasons, latency_ms, condition_note`.

### 5.8 Tests

- **pytest:**
  - the JSON evaluator equals scikit-learn's `predict_proba` to 1e-12;
  - `feat_d1` gives the same signs as the research features;
  - the golden file is deterministic;
  - the sha256 is stable;
  - the locked-test rule is deterministic, and a test person never reaches development code.
- **vitest:**
  - golden parity (D25);
  - condition blindness: random conditions, plus a `condition` getter that throws;
  - antisymmetry, using `core/mirror`;
  - refusal on a hash or version mismatch;
  - every reason code fires once;
  - display clamping;
  - the model is at most 200 KB, and the collector entry chunk is unchanged.
- **Playwright** (`?predict=1&sim=1`, Chromium and WebKit):
  - the model id is shown;
  - zero requests after load, and the CSP blocks `fetch()`;
  - an offline reload works.
- **Real devices.** At least 2 iPhones and 2 Android phones. Test airplane mode, Low Power Mode and Rotation Lock, and measure the latency (budget ≤ 50 ms, with ≤ 1 s from the end of the trial to the result).

### 5.9 The PI evaluation

**D29 PI session protocol** (replaces the attempt counts of 08 §3.1).
1. **Attempts.** 12 per person, 6 with each hand.
   - The hand order is on paper, in balanced blocks of 4 (2 left and 2 right, shuffled). The app never sees it.
   - The PI chooses the conditions. The suggested list for each hand is: upright, sideways, fast, slow, tilted touch, and one of the PI's own.
2. **Scoring.** Each slot counts its first attempt that produced a prediction; retries are allowed only for G1 and G2. The truth is entered after the prediction has been saved.
3. **Primary metric.** Forced-choice accuracy with an exact (Clopper–Pearson) 95 % CI.
4. **Secondary metrics:**
   - accuracy on answered attempts, and coverage;
   - accuracy with abstentions counted as errors;
   - Brier score (a coin scores 0.25) and log loss (a coin scores 0.693);
   - accuracy per person.

   Conditions are reported descriptively.
5. **Expectation first.** Before the session, the student commits the expectation (T15): expected forced-choice accuracy with its CI per platform and for standard vs odd movements, τ and the expected coverage. It is stored in `report/frozen/expectation_<model_id>.json` with the model's sha256. The commit time is the timestamp.
6. **What 12 attempts can show** (my computation):

   | Correct | 95 % CI | One-sided p vs chance |
   |---|---|---|
   | 12/12 | 73.5–100 % | 0.0002 |
   | 11/12 | 61.5–99.8 % | 0.003 |
   | 10/12 | 51.6–97.9 % | 0.019 |
   | 9/12 | 42.8–94.5 % | 0.073 |

   One person can show "better than chance", not a precise accuracy. Five people at 57/60 give 86.1–99.0 % (pooled, ignoring the correlation within a person).
7. **Rehearsal.** Before the PI, 3–5 people who were never in the training data run the same protocol. This is S3-M3, the "dry run", and their sessions are the stress set. They include the conditions the collector does not collect: screen away, sitting, walking, a phone case, landscape and upside down.

## 6. Report and deck

**D30 The report.**
- **Format.** KOMA-Script `scrartcl`, A4, 11 pt, Latin Modern, built with pdfLaTeX and biber through latexmk.
- **Length.** At most 20 pages of main text plus appendices A–J. The skeleton is `report/main.tex`, which compiles today with visible placeholders.
- **Sections:**
  1. Introduction
  2. Related work
  3. Data collection
  4. Sensors and implicit calibration
  5. Dataset
  6. Preprocessing and segmentation
  7. Why the hand is visible: the mirror and chirality
  8. Models
  9. Evaluation protocol
  10. Results
  11. Probability output
  12. Error analysis and ablations
  13. The predictor and the PI evaluation
  14. Limitations, ethics, reproducibility
  15. Conclusion
- **Contract.** The registry of figures (F01–F23) and tables (T01–T15), the number keys and the build steps are in [report/README.md](../report/README.md).

**D31 Generated assets.**
- **One generator, two styles.** Every data figure, data table and number is generated by `report/scripts/make_assets.py` (to be written, M6) from the runs named in `report/report.toml`. Each figure is drawn once and rendered twice:
  - a paper style: vector PDF at its final width, 8-pt text;
  - a slides style: 200-dpi PNG, 16–18-pt text.
- **Numbers in the prose** use `\R{key}`. A key with no value prints a red `??key`.
- **Git.** Generated assets are git-ignored, because the repository is public. P-codes are the only identifiers, and table cells with fewer than 3 people are suppressed.

**D32 The deck.** It has 14 slides plus 7 backups, built by `presentation/build_deck.py` (python-pptx) from [presentation/OUTLINE.md](../presentation/OUTLINE.md), the slide-style figures and `numbers.json`.
- Titles are one-sentence claims and slides use no transitions.
- The QR demo slide comes last.
- The talk takes about 13 minutes, followed by about 5 minutes of demo.

**D33 Final-mode guards.**
- **The report build aborts** on any of these:
  - a missing input or a remaining placeholder;
  - an undefined `\R` key;
  - synthetic data;
  - a name-like column (`name`, `email`, `phone`);
  - locked-test numbers from a run without `locked_test_opened_at`.
- **The deck build aborts** on unresolved `{key}` fields, missing visuals, or notes longer than 160 characters.

## 7. What we need from data collection

| Need | Minimum | Target | Why |
|---|---|---|---|
| People with tiers 0–2 complete (21 of the 29 planned trials) | 30 | 40+ | ~30 development + ~10 locked test (D3); per-person CIs |
| iPhone users | 25 % | ≥ 30 % (12+ people) | iOS sign and quantisation differ; leave-one-platform-out needs both |
| Left-handed people (self-report) | 8 | 10+ | The dominance confound (05 §8.1); only ~4 in 40 by chance, so recruit them on purpose |
| Android brands | 2 | 3+ | Device-model differences (05 §8.1) |
| Sessions per person | 1 | 1 | More people beat more trials per person (design effect, table below) |
| PI's own platform | represented | well represented | Ask the PI early which phone they will use (question a) |
| Rehearsal people (predict mode, never in training) | 3 | 5 | Stress set and dry run (D29) |

Which street conditions matter most, in order:
1. Both hands from everyone. The protocol does this by design.
2. The tier-1 speeds: slow is the hardest condition, with one slow trial per hand.
3. The tier-2 sideways grips.
4. The tier-3 variety, including the tilted touch.
5. The tier-0 choice trial.

The bonus tier is welcome but not needed. Screen-away grips are synthesised (A9), and sitting and walking come only from the rehearsal.

95 % CI half-width of accuracy, in percentage points, at 90 % accuracy with 24 instructed trials per person. The design effect is 1 + 23·ICC (my computation; re-estimate the ICC after 10 people).

| People | ICC 0.05 | ICC 0.15 | ICC 0.30 |
|---|---|---|---|
| 10 | 5.6 | 8.0 | 10.7 |
| 20 | 3.9 | 5.7 | 7.5 |
| 30 | 3.2 | 4.6 | 6.2 |
| 40 | 2.8 | 4.0 | 5.3 |

A 10-person locked test therefore resolves only about ±8 points; this is stated before it is opened. One person with 18 of 20 correct has a CI of 68–99 %, so the report shows the distribution of per-person accuracy, not individual verdicts.

## 8. Milestones and acceptance criteria

Milestone IDs map to the Stage-2 and Stage-3 IDs of [00 §3](00_PROJECT_OVERVIEW.md#3-the-three-stages).

**M0 Plan frozen (now).**
- [ ] This document is committed before the first frozen pull. Its commit fixes D3 and the pre-registered choices: the primary metric (forced-choice accuracy on unseen people), the default model family (rung 2), the gates, and the τ target.
- [ ] The run manifests record that commit hash.

**M1 Data plumbing** (before 10 people are analysed).
- [ ] Freeze step and `snapshot.json` (D1).
- [ ] `person_key`, the alias table and the duplicate candidates in QA (D2).
- [ ] The `partition='dev'` loader and the locked-test script (D3).
- [ ] `io` exposes `hand_source`, the block `kind`, `tier`, `review_mode` and `calibration.method`.
- [ ] `modeling_table` drops `hand_source = reported` by default (D4).
- [ ] The strata default is `dominant_hand × platform`.
- [ ] QA shows `implicit_axis_check` and the sign-source agreement table (D7, D8).
- **Accept when:**
  - the pilot gives the same features as before;
  - unit tests prove that a choice trial never enters training and a test person never reaches development code.

**M2 Baselines on the first 10 development people** (S2-M1, S2-M2).
- [ ] The 64-sign-pattern check (05 §3.5) runs on each platform with data.
- [ ] Rungs 0–2 run on the deploy pipeline (`feat_d1`, offline_v1, 60 Hz).
- [ ] `feature_mirror`, `MirrorSymmetrized` and the augmentation wrappers exist, with tests.
- [ ] A14 (first-half yaw vs net yaw) and the first ICC estimate are reported.
- **Accept when:**
  - shuffled labels give 45–55 %;
  - σ* = (−, +, +; +, −, −) ranks first;
  - antisymmetric models deviate from the mirror rule by at most 1e-9;
  - the record-wise vs subject-wise gap is reported.

**M3 Predictor v0** (in parallel with collection, after M2) (S3-M1, S3-M2).
- [ ] Predict mode (D22–D27) with the offline_v1 TypeScript port and `feat_d1`.
- [ ] Golden parity tests (D25) and the CSP (D28).
- [ ] The rehearsal with 3–5 new people (D29).
- **Accept when:**
  - the parity tests pass;
  - the Playwright tests show zero requests after load on Chromium and WebKit, and an offline reload works;
  - every reason code has fired once;
  - latency on a mid-range phone is at most 50 ms;
  - it works on 2 iPhones and 2 Android phones, including the check on both engines.

**M4 Model selection and freeze** (30 or more development people) (S2-M3).
- [ ] Calibration (D19) and abstention (D20).
- [ ] Rungs 3–5 where they are allowed (D14).
- [ ] Ablations A1–A14 and the learning curve.
- [ ] Selection (D15), then export of the model JSON with `registry.ts`.
- **Accept when:**
  - the chosen model's development-OOF ECE is at most 0.05 (**Rec.**);
  - the reliability diagram has no bin off by more than its CI;
  - the model sha256 is recorded.

**M5 Locked test** (40 or more people) (S2-M4).
- [ ] One run, written to its own folder, giving T10.
- [ ] The expectation T15 is generated and committed in `report/frozen/` (D29).
- **Accept when:**
  - the manifest carries `locked_test_opened_at`;
  - the numbers are reported whether or not they fall inside the development CI.

**M6 Report and deck.**
- [ ] `make_assets.py` exists, and the registry is complete.
- [ ] `latexmk` passes with `REPORT_MODE=final`, and `build_deck.py --final` passes.
- [ ] The one-page PI sheet (`report/pi_sheet.tex`): QR code, steps, scoring table, limitations.
- **Accept when** both final builds pass their guards (D33) and the report is at most 20 pages of main text.

**M7 PI session** (S3-M4).
- [ ] The protocol of D29, then T14 and F22 in the report.
- [ ] The 08 §6 hand-off checklist, as amended in §11.
- **Accept when** the session log is exported and scored against the frozen expectation.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Wrong gravity sign in predict mode | Every answer inverted | The check must agree with the engine prior, else refuse (D23); tests on both engines; the sign-agreement table (D7) |
| The §2.1 finding does not generalise across people, grips, speeds or the tilted touch | Rung 1 is weak | Rung 2 uses the whole yaw time course; A14; the learning curve; per-condition reporting |
| Too few iPhones or left-handers | Wide CIs; platform shift | Targeted recruitment (§7); leave-one-platform-out; a caveat on screen when the PI's platform is scarce |
| The PI's conditions lie outside the training data (screen away, walking, a case, landscape) | Over-confident answers | Grip invariance of chiral features (proper rotations); OOD gates; the rehearsal stress set; limitations stated in advance |
| Drift between the TypeScript and Python segmentation | Silent errors | Golden fixtures; version check; refusal on mismatch |
| Dominance shortcut | Fails for left-handers | Antisymmetric default model; A8; the dominance probe (05 §8.1) |
| Label noise from hands-free auto-accept | Lower accuracy, mis-calibration | D6 screening; flag button; sensitivity analysis |
| One person under several P-codes | Leakage, optimistic CV | D2 alias table and QA duplicate candidates |
| WebKit ignores a script-inserted CSP | A broken privacy claim (no code uploads anyway) | No uploader import; Playwright WebKit test; real iPhone check |
| Personal data in the public repository | Privacy breach | Generated assets git-ignored; P-codes only; small cells suppressed; the D33 guard |
| LaTeX toolchain on the day (PATH, Perl, missing packages) | No PDF | [report/README.md](../report/README.md) steps; the skeleton uses only installed packages |
| Time | No deep model | Deep models are optional (D14); the logistic default ships first |

## 10. Open questions for the PI

a) Which phone or phones will you test with: platform and model?
b) Forced choice only, or may the predictor say "not sure"? Do abstentions count as errors?
c) What accuracy on unseen people counts as a pass, e.g. 90 % on standard movements?
d) Will you test yourself only, or other people too, and how many attempts each?
e) Should the screen show "RIGHT 91 %" (the probability of the hand shown) or always P(right)?
f) Report: about 20 pages plus appendices, or a short 8–10-page paper? Numeric or author–year citations?
g) Present in English or in Hebrew?
h) Is a 2.5-s "hold the phone upright" check before testing acceptable? No table is needed.
i) May a user calibrate first, with a few labelled attempts? This plan assumes not.
j) Should odd movements (curved, pause, jerky, tilted touch) count in the primary score?

## 11. Proposed edits to docs/05 and docs/08

These edits are not applied here, because other workstreams own those files.

### 11.1 docs/05_MODELING_PLAN.md

1. **§1, "Which data each analysis uses".**
   - Primary row: replace "calibration present" with "gravity sign resolved from the calibration (`guided_v1` or `implicit_v1`), the trial votes or the engine, never `default`", and add "`condition.hand_source` is `instructed` or absent".
   - Add a row: "Natural use: choice trials (`hand_source = reported`); never trained on; scored out of fold; reported separately (09 D4)".
   - Add a sentence: "With `review_mode = auto`, `accepted` means completed and not redone; label noise is screened per 09 D6".
2. **§2.1.**
   - Gyro-bias row: add "Street sessions (`implicit_v1`) have no flat window; the 400 ms before GO is the only source, and every deployed model uses it (09 D9)".
   - Add a row "rotationRate axes: W3C assumed; per-session closure check flags, never remaps (09 D8)".
3. **§2.2.** Add: "Every model that ships is trained on `feat_d1` at 60 Hz (09 D11)".
4. **§2.4.**
   - Replace the third bullet with: "The deployed model uses offline_v1, identically in Python and TypeScript (09 D10). online_v1 with the predict-mode parameters re-segments 0 of 18 pilot trials, and protocol-2.0.0 natural parameters cut slow movements short (09 §2.2)".
   - Add the windows W0–W3 and the time-reversal warning (09 D11).
5. **§3.5, item 1.** Add: "Augmentation runs inside the estimator's `fit` (09 D13); for antisymmetric models it is redundant".
6. **§3.5, item 2.** Add the `MirrorSymmetrized` wrapper and `features.feature_mirror`. Replace "Calibrate with temperature scaling" with "temperature scaling fitted by inner cross-fitting (09 D19); Platt only for unconstrained models".
7. **§4.3.** Replace the paragraph "**Hyp.** Ψ > 0 for the right hand, …" with: "**Hyp.** The hand shows in the time course of ω·ĝ: in the first half of the reach the yaw is counter-clockwise seen from above for the right hand, then the wrist turns the phone back, so the net Ψ largely cancels (pilot, 16 trials, one person: first-half yaw AUC 0.95, net yaw 0.58; 09 §2.1). Rules learn their sign from data".
8. **§5.**
   - Table: rung 1 input `chi_yaw_first_half_deg`; rung 3 adds `MirrorSymmetrized(HGB)`; add rung 3b, MiniRocket (research only).
   - Augmentation table: add the row "Screen-away flip, Q = diag(−1, 1, −1): acc and gyro (−x, y, −z); for device-frame and sequence models; chiral features are invariant (unit test)".
   - Training rules: replace "Calibrate inside the training fold: temperature or Platt scaling via `CalibratedClassifierCV(cv=<group splits>)`" with the inner-cross-fitting procedure of 09 D19.
9. **§6.1.**
   - Development CV: groups `person_key`, strata `dominant_hand × platform` (the street profile has no EHI).
   - Add the rows "Leave-one-speed-out", "Leave-one-variant-out" and "Learning curve".
   - Replace the code sketch with the one in 09 §4.5.
10. **§6.2, rule 2.** Replace it with: "Augmentation happens inside the estimator's `fit`, so inner and outer validation folds hold real trials only".
11. **§6.3.** Add ECE (10 equal-mass bins) and the paired person bootstrap for differences. State that the implemented bootstrap resamples people (one level).
12. **§6.4.** Replace the first two bullets with 09 D3: 25 %, `person_key`, no salt, the student and the pilot excluded, a separate script that logs `locked_test_opened_at`. Keep "Run the test set once".
13. **§6.5.** Add item 5, "sign-source agreement (09 D7)", and item 6, "retry and abstention rates by hand (09 D5)".
14. **§7.1.** For `handedness`: use `dominant_hand` (self-report) when the Edinburgh short form is absent, as in street sessions.
15. **§7.3.** Point to the registry in `report/README.md`, with the mapping T1→T05, T2→T09, T3→T13, F1→F13, F2→F18, F3→F19, F4→F20, F5→F16, F6→F09, F7→F14, F8→F17. Add the rule: "Every number, table and data figure in the report and the deck is generated by `report/scripts/make_assets.py` from `analysis/reports/<run>/`".
16. **§8.2.** Add A9–A14 from 09 §4.7, and mark A2 (phyphox) "out of scope unless a phyphox cohort is run".
17. **§3.2.** After the sentence on OREBA, Kyritsis and OpenWatch, cite [Wang et al. 2026](https://arxiv.org/abs/2607.01376) §IV-A3, which derives the same mirror matrices independently.
18. **§11.** Replace the timeline with the milestones of 09 §8. The ethics-approval gate no longer applies: the user decided on 2026-10-24 that the volunteers are adult friends who consent in the app. Use the recruitment targets of 09 §7.
19. **§12, question 8.** Mark it answered: protocol 2.0.0 has one choice trial in tier 0.

### 11.2 docs/08_STAGE3_PREDICTOR.md

1. **§1.**
   - The entry is `?predict=1&model=<id>` (alias `?mode=predict`), at the canonical URL `https://axelfridman.github.io/nosetouch-study/?predict=1`, never a tunnel; every QR code is generated from `report/report.toml`.
   - Item 3: replace the 15-s table device check with the 2.5-s upright check of 09 D23.
   - Item 5: the result screen of 09 D27. The large number is the probability of the hand shown; colours LEFT #d1495b and RIGHT #00798c.
   - Item 6: a local log with CSV export, with the columns of 09 §5.7.
2. **§2.1.**
   - Models are bundled in the predict chunk (`app/src/predict/models/<id>.json` plus `registry.ts`); there is no runtime `models/index.json`, and the previous model stays bundled for rollback.
   - Pipeline step 5 becomes "segmentation: offline_v1, identical to training".
3. **§2.2.**
   - `PredictionV2` adds `p_left`, `status: 'ok' | 'abstain' | 'retry' | 'error'`, `reliability` (`p_lo`, `p_hi`, `n`, `k`), `model_sha256`, `latency_ms` and `versions.model_format`.
   - `createPredictor(model, { check, device })` takes the result of the check, not `SessionRecord.calibration`.
4. **§2.3.**
   - Replace the JSON sample with 09 §5.4.
   - Decision: JSON + TypeScript; ONNX only under 09 D14.
   - Add that GitHub Pages cannot send the COOP/COEP headers threads need, so onnxruntime-web would run single-threaded.
5. **§2.4.**
   - Item 3: replace the recommendation with "offline_v1 on the recorded window, in both languages (09 D10)".
   - Replace the predict-mode parameter table with 09 §5.3 and say those parameters only drive the cues.
   - Item 5: the golden fixtures are written by Python (the reference), with the contents of 09 D25.
   - Add `feat_d1` (09 D11).
6. **§2.5.** Offline row: add the CSP of 09 D28. JSON model row: bundled, at most 200 KB.
7. **§3.1.** Replace the trial counts with 09 D29: 12 attempts per person, 6 per hand, balanced blocks of 4 on paper, conditions chosen by the PI, the first attempt with a prediction counts, and a rehearsal with 3–5 new people.
8. **§3.2.**
   - Primary metric: forced-choice accuracy with an exact CI. Add log loss and the binomial note (10 of 12 needed for p < 0.05).
   - Show the development reliability diagram and ECE next to the session's Brier score and log loss.
9. **§3.3.** The expectation is generated as T15 and committed to `report/frozen/expectation_<model_id>.json` before the session.
10. **§4.**
    - Replace the table with 09 §5.6: add `sign_conflict`, `closure`, `model_integrity`, `version_mismatch`, `no_segment` and `check_failed`.
    - The calibration rows use the check.
    - The OOD check uses mirror-even descriptors only.
    - Remove the "engine prior, low confidence" fallback and the "multi-peak → medium" row.
11. **§5.** No uploads in predict mode in v1; `&log=1` is deferred (09 D28).
12. **§6.**
    - Item 2: the model lives in `app/src/predict/models/`.
    - Item 6: add the 2.5-s check on both engines.
    - Item 8: add `sign_conflict`, `unusual_movement` and `model_integrity`.
    - Item 11: the PI sheet is `report/pi_sheet.tex` (M6).
    - Item 13: rollback via `&model=`.
    - Item 15: `?predict=1&sim=1`.
13. **§7.**
    - Q4: point to 09 §10, question e.
    - Q6: replace it with "Is a 2.5-s upright check acceptable?"

## 12. Edits needed in other files (not made; owners in brackets)

- **docs/02 §12** [schema]: a note that the predictor gates G1–G4 (09 D5) are not exclusion rules.
- **docs/03 §4–5** [calibration]: describe `implicit_v1` and the upright check of predict mode.
- **README.md, docs/00 and docs/06** [docs]: "do not collect data before ethics approval" contradicts the user's decision of 2026-10-24 (no ethics review; adult volunteers who consent in the app). The report states this plainly; the documents should too.
- **analysis** [analysis]:
  - `io`: `hand_source`, `tier`, block `kind`, `review_mode`, `calibration.method`, `person_key`, `partition`.
  - `evaluate`: the `speed` and `variant` schemes, the `dominant_hand` strata default, ECE, `learning_curve`.
  - `models`: `MirrorSymmetrized`, `MirrorAugmented`, `Calibrated`.
  - New modules and scripts: `calibration.py`, `abstain.py`, `deploy.py` (`feat_d1`), `scripts/export_model.py`, the freeze script, and the run manifests with the tidy outputs of §4.8.
- **app** [app]:
  - `app/src/predict/`, the branch in `main.tsx`, and the `PredictionV2` types in `core/predictor.ts`, added without changing the existing ones.
  - `.gitattributes`: `app/src/predict/models/*.json -text`.
  - A WebKit project in `playwright.config.ts`.
