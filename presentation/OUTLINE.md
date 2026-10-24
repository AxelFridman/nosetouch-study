# NoseTouch deck: slide-by-slide outline

This is the plan for the talk to the PI and the lab, and also the source that `build_deck.py` turns into a `.pptx` ([README.md](README.md)). The deck has 14 main slides, about 13 minutes of talk followed by about 5 minutes of demo, plus 7 backup slides. The plan behind it is [docs/09 §6](../docs/09_STAGE2_3_PLAN.md#6-report-and-deck).

**Rules.**
- **Titles** are one-sentence claims of at most 2 lines and 28 pt or more.
- **Content.** One visual idea per slide and at most 3 bullets, all text 18 pt or more.
- **Figures** are rendered at their placeholder size, 200 dpi, with figure text 14 pt or more as displayed.
- **Colours.** LEFT #d1495b and RIGHT #00798c, as in the report and on the result screen.
- **Motion.** No transitions.
- **GIFs** are at most 3 MB and loop; their first frame must make sense alone, because a PDF export shows only frame 1.
- **The demo slide comes last,** so the QR code stays on screen during the demo and the questions.

**Format read by the builder.**
- `### <ID> · <title>` starts a slide, with IDs `S01`… for main slides and `B01`… for backups.
- The lines `- key: value` under it are its fields:
  - `layout`: `title`, `figure`, `two`, `demo` or `text`;
  - `message`: one bullet, repeatable;
  - `visual`: `;`-separated; each is a registry ID such as `F13` with an optional stand-in after `>`, or a path from the repository root;
  - `qr`: `small` or `large`;
  - `footer`: `yes` puts the provenance line on the slide;
  - `use_if`: a condition such as `calib/ece <= 0.05`;
  - `fallback`: the title used when that condition fails or a number is missing;
  - `notes`: one line for the speaker, at most 160 characters.
- `{key}` is filled from `report/tables/numbers.json` (the report's asset step writes it).
- Other text, like this paragraph, is ignored.

## Settings

- predictor_url: https://axelfridman.github.io/nosetouch-study/?predict=1
- numbers: report/tables/numbers.json
- figures: report/figures/slides; report/figures
- footer: {model_id} · run {run_id} · data {data_sha8}

The builder takes the predictor URL from `report/report.toml` (`[predictor].url`) when that file exists, so the report, the PI sheet and the deck share one link. The line above is only the fallback.

## Main slides

### S01 · NoseTouch: one movement tells which hand held the phone
- layout: title
- message: Arm extended, touch your nose: the phone's motion sensors alone tell left from right.
- message: {main/acc} % on people the model never saw, with a calibrated probability, offline in the browser.
- visual: docs/img/gesture/left_upright_natural.gif; docs/img/gesture/right_upright_natural.gif
- qr: small
- fallback: NoseTouch: which hand held the phone?
- notes: State the question and the answer. The code in the corner is the live predictor; we use it at the end.

### S02 · It must work for anyone, on any phone, in any grip, from one movement
- layout: two
- message: Unknown person, phone, grip and speed; the PI chooses the conditions.
- message: iPhones report acceleration with the opposite sign; browsers give only ~60 Hz.
- message: Dominant and non-dominant arms move differently: a shortcut to avoid.
- visual: docs/img/gesture/left_top_left_natural.gif; docs/img/gesture/right_angled_natural.gif
- notes: This is why "the phone moved left" cannot be the rule: grip and platform change the raw signals.

### S03 · A 5-minute street visit, nothing to install, every trial uploaded live
- layout: two
- message: Tiers put what matters first: a free-choice trial, upright, sideways, then variety.
- message: Hands-free: the app advances by itself; stopping early loses nothing.
- visual: F02 > docs/img/gesture/how_to_touch_your_nose.png; F04
- notes: Short visits still count, because the tiers put the most important conditions first.

### S04 · {data/n_people} people, {data/n_trials} trials, both hands from everyone
- layout: figure
- message: {data/ios_share} % iPhones; every person did both hands.
- message: Thin cells stated plainly: left-handers, iPhones.
- visual: F05 > analysis/reports/qa_real/counts.png
- fallback: Who and what we recorded
- notes: Point at the thin cells first: left-handers and iPhones.

### S05 · iPhones report acceleration inverted: uncorrected, a right hand looks like a left one
- layout: figure
- message: WebKit negates the accelerometer but not the gyroscope: a parity flip.
- message: A 2.5-s upright check measures the sign on every phone and must agree with the browser engine.
- visual: F06
- notes: One sign bug would have swapped every iPhone label, so the sign is measured on each phone.

### S06 · Rotation about gravity has a handedness that no grip can change
- layout: figure
- message: Mirror, any grip: acceleration (-x, y, z), rotation rate (x, -y, -z).
- message: The yaw rate about gravity flips under the mirror and is unchanged by every grip, even screen away.
- visual: F08 > docs/img/gesture/right_upright_natural_halfway.png
- notes: The mirror flips it, rotations keep it: that is the whole physics trick.

### S07 · The hand shows in how the phone turns: toward the midline first, then back
- layout: two
- message: First-half yaw separates the hands; the net yaw largely cancels.
- message: Left-hand curves are mirror images of right-hand curves.
- visual: F10; F09 > analysis/reports/baseline_synthetic/chiral_by_hand.png
- use_if: main/first_half_auc >= 0.75
- fallback: The yaw about gravity separates the hands
- notes: The pilot hinted at this; the full data decide. The 64-sign-pattern check picks the expected mirror rule.

### S08 · Physics first: ship the simplest model that is as good as the best
- layout: figure
- message: Ladder: yaw rule, antisymmetric logistic regression, boosting, sequence models.
- message: Antisymmetric means p(mirror) = 1 - p: dominance cannot be the shortcut.
- visual: T09
- footer: yes
- notes: The shipped model is the simplest one inside the best one's confidence interval: a few kilobytes of JSON.

### S09 · Tested only on people the model never saw
- layout: figure
- message: Every split is by person; a locked 25 % test set was opened once.
- message: Shuffled labels give ~50 %; mirror copies never reach a test fold.
- visual: F12
- notes: Random trial splits would inflate accuracy; ours never test on a person seen in training.

### S10 · {main/acc} % on unseen people (95 % CI {main/acc_lo}-{main/acc_hi})
- layout: figure
- message: Locked test, opened once: {locked/acc} % [{locked/acc_lo}, {locked/acc_hi}].
- message: Worst tenth of people: {main/pacc_p10} %.
- visual: F13 > analysis/reports/baseline_synthetic/roc.png
- footer: yes
- fallback: Accuracy on people the model never saw
- notes: Lead with the locked-test number, then the spread across people.

### S11 · When it says 90 %, it is right about 90 % of the time
- layout: two
- message: Calibrated on people outside the training folds: ECE {calib/ece}.
- message: Below {calib/tau} it says "not sure"; it answers {calib/coverage} % of attempts.
- visual: F16; F17
- use_if: calib/ece <= 0.05
- fallback: How far the percentages can be trusted
- footer: yes
- notes: This is what the percentage on the phone means, and why it sometimes refuses to answer.

### S12 · Where it struggles, and what helped
- layout: two
- message: Weakest conditions first: they are the ones you might test.
- message: Ablations as differences with paired confidence intervals.
- visual: F18 > analysis/reports/baseline_synthetic/subgroups.png; F15
- footer: yes
- notes: Show the weak spots first, since they are the conditions you might choose.

### S13 · What to expect when you test it
- layout: figure
- message: Expected accuracy per platform and for standard vs odd movements, committed before today.
- message: It may say "not sure" for an unusual movement or a low-confidence one.
- visual: T15
- footer: yes
- notes: I committed this before the demo, so the demo tests a prediction rather than producing a number to explain.

### S14 · Try it now: scan, hold upright, then touch your nose
- layout: demo
- message: 1. Open the link in Safari or Chrome; allow motion sensors.
- message: 2. Hold the phone upright for 2.5 s, then at arm's length in any grip.
- message: 3. At the beep, bring it to your nose; hold until the double beep.
- message: 4. Read the hand and its probability, then tap the true hand.
- visual: F21
- qr: large
- notes: You choose the hand in secret; I read the percentage aloud and log it.

## Backup slides

### B01 · The model never sees the label or the condition
- layout: text
- message: Predictor contract: sensor streams and segmentation only, never trial.condition.
- message: Tests: random conditions, a condition getter that throws, hash and version checks.
- message: The truth is entered only after the prediction is saved.
- notes: Answer to "could it be cheating?": the code cannot see the hand, and the tests prove it.

### B02 · Why 60 Hz is enough
- layout: text
- message: Movements last 1-5 s; their acceleration energy sits below a few hertz.
- message: The evidence is a rotation integral over the movement, not a fast transient.
- notes: A native app would give at most 100 Hz on iPhones and is not needed for this movement.

### B03 · Screen-away grips: a proper rotation, so chiral features do not change
- layout: figure
- message: 180° about the phone's long axis maps acceleration and rotation rate to (-x, y, -z).
- message: Checked on synthetic flips and on the rehearsal sessions.
- visual: F11
- notes: Participants could not do screen-away grips, so we proved invariance and tested it on the rehearsal set.

### B04 · Results per platform and per held-out condition
- layout: figure
- visual: F14
- footer: yes
- notes: Use when asked about iPhone vs Android or a specific grip or speed.

### B05 · Privacy: what is stored where
- layout: text
- message: The collector uploads to Heroku Postgres (EU); names stay in the database.
- message: The report and the deck use pseudonymous P-codes; small cells are suppressed.
- message: Predict mode makes no network requests at all.
- notes: The predictor you are holding cannot send anything; the policy blocks every connection.

### B06 · Anatomy of one trial
- layout: figure
- visual: F07
- notes: Onset and end from the offline segmentation; the yaw rate about gravity over the movement.

### B07 · The PI session, attempt by attempt
- layout: figure
- visual: F22
- footer: yes
- notes: Show after the demo: each attempt's probability by the true hand, against the committed expectation.
