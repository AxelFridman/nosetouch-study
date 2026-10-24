# NoseTouch deck

The talk for the PI and the lab: 14 slides plus 7 backups, about 13 minutes followed by about 5 minutes of live demo. The outline is [OUTLINE.md](OUTLINE.md), which gives each slide's title (a one-sentence claim), message, visual and speaker note. `build_deck.py` turns that outline into a `.pptx` with python-pptx. The plan behind it is [docs/09 §6](../docs/09_STAGE2_3_PLAN.md#6-report-and-deck), and the figures come from the report's asset step ([../report/README.md](../report/README.md)).

```text
presentation/
  OUTLINE.md      slide-by-slide plan; also the builder's input (format described at its top)
  build_deck.py   python-pptx builder (stub: complete layout, placeholders for missing figures)
  build/          output: NoseTouch_deck.pptx, qr_predictor.png (git-ignored)
```

## Install (once)

python-pptx and segno (for the QR code) are **not** in `analysis/requirements.txt`, and they are not installed in `analysis/.venv` yet. Add them to that venv. This changes no tracked file:

```powershell
analysis\.venv\Scripts\python -m pip install python-pptx==1.0.2 segno==1.6.6
```

Both versions installed cleanly on Python 3.11.9 in a test venv, pulling in lxml and XlsxWriter. Pillow is already in the analysis venv. Without segno, the builder draws a placeholder instead of the QR code.

## Build

From the repository root:

```powershell
analysis\.venv\Scripts\python presentation\build_deck.py              # draft -> presentation\build\NoseTouch_deck.pptx
analysis\.venv\Scripts\python presentation\build_deck.py --check      # parse and check only (works without python-pptx)
analysis\.venv\Scripts\python presentation\build_deck.py --final      # stand-ins, placeholders, missing numbers fail
analysis\.venv\Scripts\python presentation\build_deck.py --no-backup  # main slides only
```

What the builder does:

- **Visuals.** A registry ID such as `F13` resolves to the slide-style render of the report's asset step: `report/figures/slides/F13_*.png`, then `report/figures/F13_*.png`. PowerPoint cannot embed PDF figures, so PNG, GIF and JPEG are accepted.
  - Until the render exists, the outline's stand-in after `>` is used. The stand-ins are existing files such as `analysis/reports/baseline_synthetic/roc.png` and `docs/img/gesture/*.gif`.
  - If there is no stand-in either, a grey placeholder box names the ID.
  - Images keep their aspect ratio. GIFs are embedded byte for byte, so PowerPoint animates them.
- **Numbers.** `{key}` in titles and bullets is filled from `report/tables/numbers.json`.
  - A slide whose title has a `use_if` condition, e.g. `calib/ece <= 0.05`, falls back to its neutral `fallback` title when the condition fails or the number is missing.
  - A claim therefore never appears on a slide unless the data support it.
- **QR code.** It is made from `report/report.toml` `[predictor].url`, the same link as in the report and on the PI sheet, with error level M and a 4-module quiet zone. The demo slide shows it about 4.6 inches tall, so it scans from the back of a room.
- **Notes.** Every slide gets its one-line speaker note. The builder rejects missing notes and notes over 160 characters.

**Exit codes.**
- 0: built, or checked, without errors.
- 1: errors. In a draft only structural problems count, such as a missing title or note or an unknown layout. With `--final`, every warning counts too.
- 2: usage error, or python-pptx missing (the builder prints the install command).

**Verified on 2026-10-24.**
- **Build.** With python-pptx 1.0.2 and segno 1.6.6 on Python 3.11.9, the outline built into a 21-slide, 1.1 MB deck. The draft passed with 28 warnings, all of them generated figures or numbers that do not exist yet. `--final` correctly failed for the same reasons.
- **Render.** LibreOffice converted the deck to PDF, and the slides rendered as planned.
- **Without the libraries.** `--check` runs without python-pptx, and a build without it stops with the install message.

## PDF and handouts

- **PowerPoint:** File > Export > PDF.
- **LibreOffice** (close it first; the conversion uses the default profile):

  ```powershell
  & "C:\Program Files\LibreOffice\program\soffice.exe" --headless --convert-to pdf --outdir presentation\build presentation\build\NoseTouch_deck.pptx
  ```

A PDF shows only the first frame of each GIF, so every GIF must open on a representative frame.

## Demo-day checklist

- [ ] The QR code points at the GitHub Pages link, never a tunnel, and nothing was deployed in the last 48 hours.
- [ ] The code scans from the back of the room. Predict mode works in airplane mode after one load.
- [ ] Phone charged, Low Power Mode off, Rotation Lock on (iOS).
- [ ] Slide S13 (the committed expectation) is shown before the demo. The PI sheet is printed. A backup video of the demo is ready.
- [ ] `build_deck.py --final` passes, and the deck is under 50 MB.
