# NoseTouch report (LaTeX)

This folder holds the final report as a skeleton that compiles today. It contains every section with a list of the content still to write, the static tables and equations that are already final drafts, and guarded inputs for the generated numbers, tables and figures. The plan behind it is [docs/09 §6](../docs/09_STAGE2_3_PLAN.md#6-report-and-deck). The deck that reuses its figures is in [../presentation](../presentation/README.md).

```text
report/
  main.tex          the report: sections, TODO lists, static tables T01-T04, equations, guarded inputs
  references.bib    bibliography (biblatex + biber); every entry checked against its publisher or arXiv
  latexmkrc         build recipe: asset step, then pdfLaTeX + biber; auxiliary files in build/
  report.toml       settings shared with the deck: predictor URL, input runs, build mode
  figures/          generated figures (git-ignored); figures/static/ for hand-made ones (tracked)
  tables/           generated table bodies, numbers.tex, numbers.json, provenance.tex (git-ignored)
  scripts/          make_assets.py, to be written (docs/09 milestone M6)
  frozen/           the expectation committed before the PI session (docs/09 D29)
  build/, main.pdf  build output (git-ignored)
```

## Build

**Prerequisites** (checked on the development machine on 2026-10-24):

- **MiKTeX 24.1** is installed for the user at `%LOCALAPPDATA%\Programs\MiKTeX`, but it is not on PATH. Add it once in PowerShell, then open a new terminal:

  ```powershell
  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ";$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64", 'User')
  ```

- **Perl.** latexmk needs it. Either install Strawberry Perl (`winget install --id StrawberryPerl.StrawberryPerl -e`), or run the build from Git Bash, whose Perl 5.36 latexmk finds. Do not put Git's `usr\bin` on the global PATH: it hides Windows' `find` and `sort`.
- **Update MiKTeX once** (MiKTeX Console > Updates). This removes the "you have not checked for MiKTeX updates" message.
- **Optional packages.** `bookmark` and `csquotes` are loaded automatically when they are installed. siunitx, cleveref, threeparttable, makecell, qrcode and xurl are not installed yet. If later sections use them, MiKTeX's on-the-fly installer fetches them on the first build with internet, or install them in advance with `miktex packages install <name>`. The skeleton itself uses only packages that are already installed.

**Commands**, from this folder:

```bash
latexmk                       # main.pdf; auxiliary files in build/
latexmk -pvc                  # rebuild on every save
latexmk -C                    # remove everything latexmk made
REPORT_SKIP_ASSETS=1 latexmk  # Git Bash: build without the asset step
```

In PowerShell, set the mode first: `$env:REPORT_MODE='final'; latexmk` builds the release with the guards on.

**Verified on 2026-10-24.**
- The skeleton compiled with MiKTeX 24.1 (pdfTeX 1.40.25, biber) through latexmk 4.88 from Git Bash, with MiKTeX's package installer disabled: 17 pages, about 26 s cold.
- The log contains no errors, no undefined citations or references, and no overfull boxes.
- The remaining warnings are expected: undefined `\R` keys (no results yet), and the hints that `bookmark` and `csquotes` are missing.

## How numbers, tables and figures get into the PDF

The asset step `scripts/make_assets.py --config report.toml --mode draft|final` is to be written (milestone M6). `latexmk` runs it first when it exists. It reads only the analysis run folders named in `report.toml` (`[runs]`), which follow the contract in [docs/09 §4.8](../docs/09_STAGE2_3_PLAN.md#48-what-each-analysis-run-must-write-for-the-report). It writes:

| Output | Content | Used by |
|---|---|---|
| `tables/numbers.tex`, `tables/numbers.json` | One value per key, e.g. `\expandafter\def\csname res@main/acc\endcsname{91.2}`. A key with a CI also has `_lo` and `_hi`. All formatting lives here: percentages with 1 decimal, probabilities with 2, counts as integers. | `\R{key}` and `\Rci{key}` in `main.tex`; `{key}` in the deck |
| `tables/provenance.tex` | `\def\datakind{real}` (or `synthetic`), `\def\buildlabel{...}`, `\def\provenanceline{git <hash> · data <sha8> · model <id>}`, `\def\predictorURL{...}`, and `\finalbuildtrue` in final mode | the header, footer, watermark and guards |
| `tables/<ID>_<slug>.tex` | Only the `tabular`: booktabs, no vertical rules, CIs as `[lo, hi]`. The caption and label stay in `main.tex`. Every string from a record is escaped, because records come from a public endpoint. | `\gentab{ID}{<ID>_<slug>}` |
| `figures/<ID>_<slug>.pdf` | Paper style. Vector, drawn at its final width (6.2 in or 3.0 in) and included at scale 1, so all text is 8-pt Latin Modern. Saved with `savefig(..., metadata={'CreationDate': None})`, so an unchanged figure does not force another LaTeX pass. | `\genfig{ID}{<ID>_<slug>}` |
| `figures/slides/<ID>_<slug>.png` | Slides style: the same figure at 200 dpi with 16–18-pt text | `presentation/build_deck.py` |

Until these files exist:
- `\R` prints a red `??key` and logs a warning;
- `\gentab` and `\genfig` print a grey box naming the missing file;
- every page carries the watermark "NO RESULTS YET", which becomes "SYNTHETIC DATA" when `\datakind` is `synthetic`.

Number keys used so far:
- `data/`: `n_people`, `n_trials`, `n_ios`, `n_android`, `ios_share`;
- `main/`: `acc` (with `_lo` and `_hi`), `auc`, `pacc_p10`;
- `locked/`: `acc` (with `_lo` and `_hi`), `n_people`;
- `calib/`: `ece`, `tau`, `coverage`.

Keys are named `<scope>/<metric>`, with the scopes `data`, `main`, `locked`, `calib`, `abl` (ablations) and `pi` (the PI session).

## Registry of figures and tables

The IDs are stable. The report, the deck and docs/09 all refer to them. The brackets give the ID in docs/05 §7.3.

| ID | Shows | File stem | Source (in `analysis/reports/<run>/`) | Report | Slide |
|---|---|---|---|---|---|
| F01 | Gesture frames, cue track, device axes | static frames now | `docs/img/gesture/` and one trial | §3 | S01, S06 |
| F02 | Three app screens | `F02_app_screens` | Playwright screenshots in `?sim=1` (silent) | §3 | S03 |
| F03 | System overview | TikZ in `main.tex` | – | §3 | – |
| F04 | Street tiers: planned vs completed | `F04_tier_completion` | `tier_completion.csv` (new) | §3 | S03 |
| F05 | Coverage of grip × speed × hand | `F05_coverage` | QA `tables/accepted_trials_*_x_hand.csv` | §5 | S04 |
| F06 | Ψ before vs after normalisation, by platform | `F06_sign_normalisation` | `features.csv` (+ un-normalised Ψ) | §4 | S05 |
| F07 | Anatomy of one trial | `F07_trial_anatomy` | `plot_trial.py` data | §6 | B6 |
| F08 | Top-view chirality schematic | `F08_chirality_schematic` | static (TikZ) | §7 | S06 |
| F09 [F6] | First-half yaw by hand × platform | `F09_first_half_yaw` | `features.csv`, `trials_meta.csv` | §7 | S07 |
| F10 | ω·ĝ over normalised time, left vs mirrored right | `F10_yaw_timecourse` | `yaw_timecourse.csv` (new) | §7 | S07 |
| F11 | 64-sign-pattern check | `F11_sign_patterns` | `sign_patterns.csv` (new) | App. F | B3 |
| F12 | Split design | `F12_split_design` | static | §9 | S09 |
| F13 [F1] | Per-person accuracy | `F13_participant_accuracy` | `participant_accuracy.csv` | §10 | S10 |
| F14 [F7] | Accuracy by held-out grip, platform, speed, variant | `F14_robustness` | `metrics.csv`, `subgroups.csv` | §10 | B4 |
| F15 | Ablations: Δ with paired CIs | `F15_ablations` | `ablations.csv` (new) | §12 | S12 |
| F16 [F5] | Reliability diagram, overall and per platform | `F16_reliability` | `oof_<scheme>.csv` | §11 | S11 |
| F17 [F8] | Accuracy vs coverage, τ marked | `F17_risk_coverage` | `oof_<scheme>.csv` | §11 | S11 |
| F18 [F2] | Grip × speed accuracy with n | `F18_grip_speed` | OOF + `trials_meta.csv` | §12 | S12 |
| F19 [F3] | Accuracy vs measured duration | `F19_duration` | OOF + `trials_meta.csv` | §12 | – |
| F20 [F4] | Mixed-model odds ratios (optional) | `F20_glmm` | `glmm.csv` | App. F | – |
| F21 | Predictor pipeline and result screen | `F21_predictor_pipeline` | static + screenshot | §13 | S14 |
| F22 | PI session: p_right per attempt by true hand | `F22_pi_session` | `pi_session.csv` | §13 | B7 |
| F23 | ROC curves | `F23_roc` | OOF | App. F | – |

Tables:
- **Static,** already in `main.tex`: T01 prior work, T02 street protocol, T03 platforms, T04 mirror and screen-away rules, plus the notation table.
- **Generated:**
  - T05 [T1] participants, `T05_participants`;
  - T06 exclusions, `T06_exclusions`;
  - T07 QA and calibration per platform, `T07_qa_calibration`;
  - T08 models, `T08_models`;
  - T09 [T2] main results, `T09_main_results`;
  - T10 locked test, `T10_locked_test`;
  - T11 ablations, `T11_ablations`;
  - T12 calibration, `T12_calibration`;
  - T13 [T3] subgroups, `T13_subgroups`;
  - T14 PI session, `T14_pi_session`;
  - T15 expectation, `T15_expectation`;
  - the appendix tables TD feature dictionary, TE segmentation parameters and TG model card.

## Draft and final builds

| | `draft` (default) | `final` (`REPORT_MODE=final`) |
|---|---|---|
| Missing input | grey box, red `??key` | the build stops |
| TODO list left | shown with orange bullets | the build stops (`\finalbuildtrue` makes `todolist` an error) |
| Synthetic data | "SYNTHETIC DATA" watermark | the build stops |
| Name-like columns (`name`, `full_name`, `email`, `phone`) in any input | warning | the build stops |
| Locked-test numbers from a run without `locked_test_opened_at` | warning | the build stops |

## Style

- **Captions** state the claim first, then what is shown, n, and how the CI was computed.
- **Tables** use booktabs with no vertical rules. CIs go in brackets, and a best row is bold only when its CI excludes the others.
- **Hand colours** are LEFT #d1495b and RIGHT #00798c (`motionlab.plots.HAND_COLORS`), always paired with a marker or line style so a greyscale print still works.
- **Numbers** in the text come only from `\R`. Never use `\R` in headings.
- **Notation macros** (`\vomega`, `\ghat`, `\Smat`, `\pright`, …) live in the preamble of `main.tex`.

## Privacy

The repository is public, so generated figures and tables stay out of git (`.gitignore`). The report and the deck use pseudonymous P-codes only, and table cells with fewer than 3 people are suppressed. Names stay in the database. Predict mode makes no network requests.

## Troubleshooting

| Message | Fix |
|---|---|
| `MiKTeX could not find the script engine 'perl'` | Install Strawberry Perl, or build from Git Bash. |
| `pdflatex: major issue: So far, you have not checked for MiKTeX updates.` | Harmless. Run the MiKTeX update once. |
| `MiKTeX requires Windows 10 (or greater)` from some tools | Old 24.1 binaries; harmless. The update removes it. |
| A missing `.sty` file | MiKTeX installs it on the fly with internet access, or `miktex packages install <name>`. |
| Warnings about `bookmark`, `csquotes` or the caption package's "unknown document class" | Harmless with the skeleton. Installing `bookmark` and `csquotes` silences the first two. |
