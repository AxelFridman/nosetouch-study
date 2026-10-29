"""One figure style for every Stage 2 figure, and ``save_fig`` (vector PDF + PNG preview + caption registry).

Style (report/README.md: figures are drawn at their final width and included at scale 1):
* widths ``FULL_W`` = 6.2 in (text width) and ``HALF_W`` = 3.0 in; 8-pt text in Latin Modern Roman (the
  report's font, registered from MiKTeX when installed; DejaVu Serif otherwise), Computer Modern math;
* colour carries one job per figure. Hands: left ``#D1495B``, right ``#00739A`` (the report's handleft and
  a re-stepped handright: the report's ``#00798C`` fails the chroma floor and sits in the CVD warn band
  against the left colour; ``#00739A`` keeps the teal hue and passes every check of the dataviz
  validator with CVD dE 9.7 under protanopia and 27.5 in normal vision; update
  ``\\definecolor{handright}`` in report/preamble.tex to match). Platforms: Android ``#008300``, iOS ``#6250d6``
  (CVD dE 28.8). Ordered levels (speed, tier): one blue ramp, light -> dark. Never more hues than needed;
  identity is also given by a legend or a direct label;
* thin marks, solid hairline grid, no top / right spines;
* ``save_fig(fig, name, caption=..., decision=...)`` writes figures/<name>.pdf (CreationDate stripped, so
  an unchanged figure is byte-identical) and figures/<name>.png (200 dpi, no software stamp), records the
  caption and the decision the figure justifies in figures/captions.json, and every string drawn in the
  figure in figures/figure_texts.json (scanned by the privacy test). Both registries are updated under a
  cross-process lock and replaced atomically (``stage2.locking``), so parallel processes never drop an entry.
  The images are rendered to temporary files first and moved into place under the same lock, together with the
  registry update: a reader holding the folder's lock (``captions.json.lock``) always sees every PDF with its
  registry entries, and a reader without it never sees a half-written image.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from . import config as C  # noqa: E402
from .locking import atomic_path, file_lock, replace_file  # noqa: E402

FULL_W = 6.2
HALF_W = 3.0

HAND_COLORS = {'left': '#D1495B', 'right': '#00739A'}
HAND_LABELS = {'left': 'Left hand', 'right': 'Right hand'}
PLATFORM_COLORS = {'android': '#008300', 'ios': '#6250d6'}
PLATFORM_LABELS = {'android': 'Android', 'ios': 'iOS'}
RAMP = ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b')   # one-hue ordinal ramp (validated)
SPEED_COLORS = {'slow': RAMP[0], 'natural': RAMP[2], 'fast': RAMP[4]}
TIER_COLORS = {0: RAMP[0], 1: RAMP[1], 2: RAMP[2], 3: RAMP[3], 4: RAMP[4]}
SERIES = '#2a78d6'          # a single neutral series
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
EVENT_COLORS = {'go': '#52514e', 'onset': '#0b0b0b', 'end': '#0b0b0b', 'contact': '#eda100',
                'cue': '#898781', 'return': '#898781'}

_STYLE_DONE = False


def _register_latin_modern() -> str | None:
    base = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'MiKTeX' / 'fonts' / 'opentype' / 'public' / 'lm'
    files = ['lmroman10-regular.otf', 'lmroman10-italic.otf', 'lmroman10-bold.otf', 'lmroman10-bolditalic.otf']
    found = [base / f for f in files if (base / f).exists()]
    if not found:
        return None
    logging.getLogger('fontTools').setLevel(logging.ERROR)  # OTF 'created timestamp' notices
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for f in found:
            try:
                font_manager.fontManager.addfont(str(f))
            except (OSError, RuntimeError, ValueError):
                return None
    return 'Latin Modern Roman'


def use_style() -> None:
    """Apply the Stage 2 rcParams once (idempotent)."""
    global _STYLE_DONE
    if _STYLE_DONE:
        return
    fam = _register_latin_modern()
    plt.rcParams.update({
        'font.family': [fam, 'DejaVu Serif'] if fam else ['DejaVu Serif'],
        'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'axes.unicode_minus': True,
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'figure.titlesize': 9,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
        'axes.axisbelow': True, 'lines.linewidth': 1.2, 'lines.solid_capstyle': 'round',
        'legend.frameon': False, 'figure.dpi': 100, 'savefig.dpi': 200, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.prop_cycle': matplotlib.cycler(color=[SERIES, '#eb6834', '#1baf7a']),
    })
    _STYLE_DONE = True


def figure(width: str | float = 'full', height: float = 2.4, nrows: int = 1, ncols: int = 1, **kw: Any):
    """(fig, axes) with the Stage 2 style; width 'full' (6.2 in), 'half' (3.0 in) or inches."""
    use_style()
    w = FULL_W if width == 'full' else (HALF_W if width == 'half' else float(width))
    kw.setdefault('layout', 'constrained')
    return plt.subplots(nrows, ncols, figsize=(w, height), **kw)


def plain_log_ticks(axis, ticks) -> None:
    """Plain-number tick labels on a log axis (0.3, 1, 3 instead of 3x10^-1)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def hand_handles(hands: tuple[str, ...] = ('left', 'right'), **line_kw: Any) -> list[Any]:
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=HAND_COLORS[h], lw=line_kw.get('lw', 1.6), label=HAND_LABELS[h]) for h in hands]


def _load_captions(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}
    return {}


def figure_texts(fig: Any) -> list[str]:
    """Every string drawn in a figure (titles, labels, tick labels, legends, annotations), after a draw."""
    from matplotlib.text import Text
    return sorted({t.get_text() for t in fig.findobj(Text) if t.get_text().strip()})


def save_fig(fig: Any, name: str, *, caption: str, decision: str = '', fig_dir: str | Path | None = None,
             png_dpi: int = 200) -> dict[str, str]:
    """Write <fig_dir>/<name>.pdf and .png, register the caption (figure name -> caption, decision) and
    record every string drawn in the figure in figure_texts.json (scanned by the privacy test: a PDF stores
    text as glyph ids, so a byte search of the file would miss a name)."""
    d = Path(fig_dir) if fig_dir is not None else C.FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    pdf, png = d / f'{name}.pdf', d / f'{name}.png'
    tag = f'{os.getpid()}.{uuid.uuid4().hex[:12]}'
    tmp_pdf, tmp_png = d / f'.{pdf.name}.{tag}.tmp', d / f'.{png.name}.{tag}.tmp'   # not matched by *.pdf / *.png
    cap_path = d / 'captions.json'
    txt_path = d / 'figure_texts.json'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig.savefig(tmp_pdf, format='pdf', metadata={'CreationDate': None, 'Producer': None}, bbox_inches=None)
            fig.savefig(tmp_png, format='png', dpi=png_dpi, metadata={'Software': None})
        texts = figure_texts(fig)
        plt.close(fig)
        # both registries of the folder are read-modify-write: one cross-process lock guards them, and each file is
        # replaced atomically, so figures saved by parallel processes into one folder never drop an entry; the images
        # are moved into place under the same lock, so a locked reader sees images and registries agree
        with file_lock(cap_path):
            replace_file(tmp_pdf, pdf)
            replace_file(tmp_png, png)
            caps = _load_captions(cap_path)
            caps[name] = {'caption': caption, 'decision': decision, 'pdf': pdf.name, 'png': png.name}
            with atomic_path(cap_path) as tmp:
                tmp.write_text(json.dumps(dict(sorted(caps.items())), indent=1, ensure_ascii=False), encoding='utf-8')
            reg = _load_captions(txt_path)
            reg[name] = texts
            with atomic_path(txt_path) as tmp:
                tmp.write_text(json.dumps(dict(sorted(reg.items())), indent=1, ensure_ascii=False), encoding='utf-8')
    finally:
        for t in (tmp_pdf, tmp_png):
            if t.exists():
                try:
                    t.unlink()
                except OSError:
                    pass
    return {'pdf': str(pdf), 'png': str(png)}
