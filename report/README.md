# NoseTouch report (LaTeX)

"Which hand? Detecting left- vs right-hand nose touches from smartphone motion sensors" (Axel Fridman; PI: Izhar Bar-Gad). The report comes in three versions that share one preamble and one set of generated numbers, figures and tables: the full report (`main.pdf`), a short version of about 15 pages (`short.pdf`) and a brief version of 3-4 pages (`brief.pdf`). Each version is a driver (`main.tex`, `short.tex`, `brief.tex`) and one file per section. Every number, figure and table comes from the saved Stage 2 analysis outputs (`analysis/reports/stage2/`, git-ignored) through the asset step; nothing here fits a model.

```text
report/
  preamble.tex        shared preamble: packages, number macros, \genfig, \gentab, \pending, colours, layout, notation
  main.tex            full report driver: \documentclass, \input{preamble}, title, \InputSection of every section
  sections/           00_abstract, 01_introduction ... 08_discussion (main text); A_data ... I_case_study (appendix)
  short.tex, short/   short version (about 15 pages): 00_abstract ... 08_discussion; short/diagram_pipeline.tex is
                      the overview diagram (TikZ), shared with the brief version
  brief.tex, brief/   brief version (3-4 pages, two columns): 00_abstract, 01_question ... 05_discussion
  references.bib      bibliography (biblatex + biber)
  latexmkrc           build recipe: asset step, then pdfLaTeX + biber; auxiliary files in build/
  report.toml         where the analysis outputs are ([runs] stage2), the cache metadata, the predictor link
  scripts/make_assets.py   the asset step (copies outputs, writes numbers.tex and provenance.tex, checks usage)
  figures/<unit>/     generated figures + captions.json (git-ignored); core/ holds the describe figures S2_*
  figures/photo/      hand-placed session photo (git-ignored; the asset step never touches it)
  figures/static/     hand-made figures (tracked; never touched by the asset step)
  tables/<unit>/      generated table floats; tables/numbers.tex, numbers_index.tsv, provenance.tex (git-ignored)
  build/              auxiliary build output (git-ignored)
  main.pdf, short.pdf, brief.pdf   the three compiled versions (tracked)
```

## Build

From Git Bash (MiKTeX is not on its PATH; latexmk needs Git's Perl):

```bash
export PATH="/c/Users/fridm/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd report
latexmk                         # asset step (once) + main.pdf, short.pdf, brief.pdf (about 2 min cold: ~26,000 keys)
latexmk short.tex               # one version only (also brief.tex, main.tex)
REPORT_SKIP_ASSETS=1 latexmk    # LaTeX only
REPORT_MODE=final latexmk       # final build: a missing asset, ??key or \pending stops the build
latexmk -C                      # remove everything latexmk made
```

latexmk builds `main.tex` first: the short and brief versions read its labels (`build/main.aux`) to point into the full report.

Or from `analysis/`: `.venv/Scripts/python.exe -m stage2.pipeline report` (asset step, then latexmk; `--no-latex` for the asset step only). `python -m stage2.pipeline all` ends with the same step.

## The three versions

| | Full (`main.tex`) | Short (`short.tex`) | Brief (`brief.tex`) |
|---|---|---|---|
| Length | 24 pages of main text + appendices A-I | about 15 pages with figures and references | 3-4 pages |
| Layout | A4, 11 pt, one column | as the full report | A4, 10 pt, two columns |
| Sections | `sections/` | `short/` | `brief/` |
| Figures | all | about 12 essential ones, at their natural width | 3 generated (`figure*`, across both columns) + the diagram |
| Details | appendices | the full report, via `\fullref{label}` | the full report, via `\fullref{label}` |

- The three versions share `preamble.tex`, so every macro, colour and number works the same in each; a driver sets only its `\documentclass` options, its title and `\sectiondir` (the folder `\InputSection` reads).
- `\fullref{label}` prints the number of a section, figure or table of the full report and links to `main.pdf` (package `xr-hyper`): `full report, Appendix~\fullref{app:final}`.
- The short and brief versions say the same as the full report, with fewer words: one or two sentences per paragraph, captions of one or two sentences. Content changes go to the full report first.
- Generated figures are drawn 6.2 in wide (the full report's text width). Use them at their natural size: in the brief version only as `figure*`, never shrunk into one column. The overview diagram `short/diagram_pipeline.tex` is drawn for one column.
- `brief.tex` defines `\nosectionbarrier` before the preamble, so its floats may cross a section; the other two keep floats in their section.

## How numbers, tables and figures get into the PDF

`scripts/make_assets.py --config report.toml --mode draft|final` (latexmkrc runs it first) reads only `[runs] stage2` and writes, atomically and only when a file changed:

| Output | From | Used in the text as |
|---|---|---|
| `figures/<unit>/<NAME>.pdf` (PNG fallback), `figures/<unit>/captions.json` | every `captions.json` under `analysis/reports/stage2/figures/` (root registry -> `core/`) | `\genfig{<unit>/<NAME>}` inside your own `figure` float |
| `tables/<unit>/<name>.tex` | every `.tex` under `analysis/reports/stage2/tables/<unit>/`; each tabular wrapped in `\fitwidth` | `\gentab{<unit>/<name>}`: a complete float with its caption and `\label{tab:<name>}` |
| `tables/numbers.tex`, `tables/numbers_index.tsv` | every `results/<unit>_numbers.json` | `\R{<unit>/<key>}`, `\Rb`, `\Rci`, `\Rcib` |
| `tables/provenance.tex` | git, the cache's `meta.json`, `describe` numbers | page footer: git hash, data identity, persons, trials |

Number macros:
- `\R{final/some_key}` prints the value as its numbers file formats it, **with** its unit sign: a `.1%` entry prints `89.3 %`. Never add `\%` after `\R`.
- `\Rb{...}` prints the bare number (`89.3`): use it for percentage-point differences (`\Rb{...}~points`).
- `\Rci{key}` prints `value [key_lo, key_hi]` (bounds bare); `\Rcib{key}` the same without the unit sign.
- Find a key: `grep -i "calibration slope" tables/numbers_index.tsv` (key, printed value, description, source).
- A key that does not exist prints a red `??key`; `python scripts/make_assets.py --check` lists every missing key, figure and table used in the drivers and in `sections/`, `short/` and `brief/` (exit 1 in final mode).

A unit whose outputs appear later (for example `demo`) is picked up by the next run of the asset step.

## Change a figure and rebuild

1. Edit the presentation defaults (`FIGURES` dict: sizes, labels, ticks) in `analysis/stage2/lanes/<unit>_figures.py` (core describe figures: `analysis/stage2/scripts/describe_figures.py`). Never touch result logic.
2. `cd analysis && .venv/Scripts/python.exe -m stage2.pipeline figures --lanes <unit> [--only <NAME>]` (seconds; no model is fitted).
3. `cd report && latexmk` (the asset step copies the new file).

Tables: `python -m stage2.pipeline tables --lanes <unit>`; numbers: `python -m stage2.pipeline numbers --lanes <unit>`.

## Writing rules (summary; the full style guide is in the report plan)

- Persons are P01..P20 (PD01 only in the case study). No names, participant codes, session ids, timestamps or device serials.
- Every result number through `\R`-macros; every figure through `\genfig`, every table through `\gentab`. Refer to each in the text.
- Evaluation is leave-one-person-out everywhere. 95 % CIs are person-cluster bootstrap intervals.
- Natural use is right-hand recall on the free-choice trials (all right-hand so far).
- `\pending{...}` marks content still to write; a final build stops on it.

## Draft and final builds

| | `draft` (default) | `final` (`REPORT_MODE=final`) |
|---|---|---|
| Missing figure, table or number used in the text | grey box, red `??key` | the asset step stops |
| `\pending` or a `todolist` left | orange text | LaTeX stops |
| Registered figure without its file | warning | the asset step stops |

## Privacy

The repository is public: generated figures and tables, the photo and the build output stay out of git (`.gitignore`). Report-facing tables and figure data are checked by the analysis' privacy tests (no name tokens, codes, backend ids or timestamps).

## Troubleshooting

| Message | Fix |
|---|---|
| `MiKTeX could not find the script engine 'perl'` | Build from Git Bash, or install Strawberry Perl. |
| `So far, you have not checked for MiKTeX updates` | Harmless. |
| `Command '\Rn' already defined` | biblatex owns `\Rn` (Roman numerals); the bare-number macro is `\Rb`. |
| A table runs off the page | It is shrunk to the text width by `\fitwidth`; a very long table needs splitting or the appendix. |
