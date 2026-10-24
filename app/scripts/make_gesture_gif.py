"""Assembles the gesture illustration frames into images for the docs (docs/img/gesture/).

Called by scripts/render_gesture_images.ts, which renders the frames in a headless browser:

  gif    PNG frames (frame_000.png, frame_001.png, ...) -> one looping GIF. The frames are optionally
         downscaled (--scale), quantised to ONE shared palette without dithering (flat artwork): the
         drawing's token colours (--keep: the touch burst, its halo, both sleeves, the line art, the
         screen, ...) are reserved exactly and the rest of the palette adapts to ALL frames, so a colour
         that appears in a few frames only (the burst at the touch) is never lost. Identical consecutive
         frames are merged, the per-frame delays are whole centiseconds that add up to the exact loop
         length, and the GIF loops forever. If the file is larger than --max-bytes it is redone with
         fewer colours, then smaller. --require checks the written GIF: every listed colour must be
         present (e.g. the burst), else the script fails.
  sheet  the three-step strip (start -> halfway -> touching) -> "how to" contact sheet: a title and one
         caption per step above the strip, centred on the given column positions.
  png    re-save a PNG with lossless optimisation (key frames).

Usage (from app/):
  python scripts/make_gesture_gif.py gif --frames DIR --out FILE.gif --fps 20 [--scale 1] [--max-bytes 2500000]
         [--keep "#e08600,#0f172a,..."] [--require "#e08600"]
  python scripts/make_gesture_gif.py sheet --strip STRIP.png --out FILE.png --centers 210,640,1070
         --captions "1|2|3" [--title TEXT] [--note TEXT]
  python scripts/make_gesture_gif.py png --src IN.png --out OUT.png
Needs Pillow (the analysis venv has it: analysis/.venv). Output is deterministic for the same frames.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (245, 247, 251)  # --bg of the light theme
INK = (15, 23, 42)  # --text
MUTED = (88, 97, 121)  # --muted

# Token colours of the light-theme drawing (gesture.css, styles.css, session.css): reserved in the palette.
KEEP_DEFAULT = ','.join(
    [
        '#e08600',  # the touch burst
        '#0f172a',  # its halo, the outlines and the eye
        '#c2410c', '#6d28d9', '#0e7490',  # right / left sleeve, participant's choice
        '#ffffff', '#f5e2d4', '#cdd6e8',  # skin, your nose, shirt
        '#eef1f7', '#f5f7fb',  # stage, page
        '#e6eeff', '#2f5bea', '#1e293b', '#475569',  # screen, "top" arrow, phone, back of the phone
        '#586179',  # the dashed path
    ]
)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.strip().lstrip('#')
    if len(c) == 3:
        c = ''.join(ch * 2 for ch in c)
    if len(c) != 6:
        raise SystemExit(f'bad colour {c!r}')
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _colours(arg: str) -> list[tuple[int, int, int]]:
    return [_hex(c) for c in arg.split(',') if c.strip()]


# ───────────────────────────── GIF ─────────────────────────────


def _load_frames(frames_dir: Path, scale: float) -> list[Image.Image]:
    files = sorted(frames_dir.glob('frame_*.png'))
    if not files:
        raise SystemExit(f'no frame_*.png in {frames_dir}')
    frames = []
    for f in files:
        with Image.open(f) as im:
            rgb = im.convert('RGB')
        if scale != 1:
            w, h = rgb.size
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        frames.append(rgb)
    return frames


def _durations_ms(n: int, fps: float) -> list[int]:
    """Whole-centisecond delays (GIF resolution) whose sum is the exact loop length n / fps."""
    edges = [round(i * 100 / fps) for i in range(n + 1)]
    return [max(1, b - a) * 10 for a, b in zip(edges, edges[1:])]


def _palette(frames: list[Image.Image], colors: int, keep: list[tuple[int, int, int]]) -> Image.Image:
    """One palette for every frame: the reserved colours exactly, the rest adapted to ALL frames."""
    w, h = frames[0].size
    # every frame counts (a mosaic of thumbnails), so a colour seen in a few frames still gets an entry
    tw, th = max(1, w // 4), max(1, h // 4)
    cols = 8
    rows = (len(frames) + cols - 1) // cols
    mosaic = Image.new('RGB', (tw * cols, th * rows), BG)
    for i, fr in enumerate(frames):
        mosaic.paste(fr.resize((tw, th), Image.Resampling.NEAREST), ((i % cols) * tw, (i // cols) * th))
    adaptive = max(2, colors - len(keep))
    q = mosaic.quantize(colors=adaptive, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = list(keep)
    pal = q.getpalette()[: adaptive * 3]
    for i in range(0, len(pal) - 2, 3):
        c = (pal[i], pal[i + 1], pal[i + 2])
        if c not in flat:
            flat.append(c)
    flat = flat[:256]
    img = Image.new('P', (1, 1))
    img.putpalette([v for c in flat for v in c] + [0, 0, 0] * (256 - len(flat)))
    return img


def _quantise(frames: list[Image.Image], colors: int, keep: list[tuple[int, int, int]]) -> list[Image.Image]:
    palette = _palette(frames, colors, keep)
    return [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]


def _check_required(out: Path, required: list[tuple[int, int, int]], min_pixels: int = 12) -> None:
    """Every required colour must appear (at least min_pixels) in some frame of the written GIF."""
    if not required:
        return
    found = {c: 0 for c in required}
    with Image.open(out) as im:
        for i in range(getattr(im, 'n_frames', 1)):
            im.seek(i)
            for count, c in im.convert('RGB').getcolors(maxcolors=1 << 22) or []:
                if c in found:
                    found[c] = max(found[c], count)
    missing = [f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c, n in found.items() if n < min_pixels]
    if missing:
        raise SystemExit(f'{out.name}: colour(s) missing from every frame: {", ".join(missing)}')


def _write_gif(frames: list[Image.Image], durations: list[int], out: Path) -> None:
    # merge identical consecutive frames (rests and holds) into one longer frame
    merged: list[Image.Image] = []
    merged_ms: list[int] = []
    last = None
    for fr, ms in zip(frames, durations):
        data = fr.tobytes()
        if last is not None and data == last:
            merged_ms[-1] += ms
            continue
        merged.append(fr)
        merged_ms.append(ms)
        last = data
    merged[0].save(
        out,
        save_all=True,
        append_images=merged[1:],
        duration=merged_ms,
        loop=0,
        optimize=True,
        disposal=1,
    )


def make_gif(
    frames_dir: Path,
    out: Path,
    fps: float,
    scale: float,
    max_bytes: int,
    keep: list[tuple[int, int, int]],
    required: list[tuple[int, int, int]],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = None
    for size_factor in (1.0, 0.9, 0.8, 0.7):
        frames = _load_frames(frames_dir, scale * size_factor)
        durations = _durations_ms(len(frames), fps)
        for colors in (128, 96, 64, 48):
            _write_gif(_quantise(frames, colors, keep), durations, out)
            size = out.stat().st_size
            if size <= max_bytes:
                _check_required(out, required)
                w, h = frames[0].size
                print(f'{out.name}: {w}x{h}px, {len(frames)} frames @ {fps:g} fps, {colors} colours, {size / 1024:.0f} KB')
                return
    _check_required(out, required)
    print(f'warning: {out.name} is still {out.stat().st_size / 1024:.0f} KB (> {max_bytes / 1024:.0f} KB)', file=sys.stderr)


# ───────────────────────────── contact sheet ─────────────────────────────


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """A scalable font: DejaVu (bundled with matplotlib), the OS fonts, else Pillow's own."""
    names = ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
    try:
        import matplotlib

        names.insert(0, str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / names[0]))
    except Exception:
        pass
    windir = os.environ.get('WINDIR', 'C:/Windows')
    names += [
        f'{windir}/Fonts/segoeuib.ttf' if bold else f'{windir}/Fonts/segoeui.ttf',
        f'{windir}/Fonts/arialbd.ttf' if bold else f'{windir}/Fonts/arial.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split('\n'):
        line = ''
        for word in para.split():
            trial = f'{line} {word}'.strip()
            if draw.textlength(trial, font=font) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def make_sheet(strip: Path, out: Path, centers: list[float], captions: list[str], title: str, note: str) -> None:
    with Image.open(strip) as im:
        body = im.convert('RGB')
    w, h = body.size
    unit = max(1.0, w / 900)  # type sizes follow the strip width
    pad = round(24 * unit)
    title_font = _font(round(30 * unit), bold=True)
    cap_font = _font(round(19 * unit), bold=True)
    note_font = _font(round(16 * unit))
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    # a caption may be as wide as the distance between neighbouring columns (minus a gap)
    spacing = min((b - a for a, b in zip(centers, centers[1:])), default=w)
    col_w = round(min(spacing, w) - 20 * unit)
    cap_lines = [_wrap(probe, c, cap_font, col_w) for c in captions]
    line_h = round(cap_font.size * 1.3) if hasattr(cap_font, 'size') else round(24 * unit)
    title_h = round(title_font.size * 1.5) if title and hasattr(title_font, 'size') else 0
    caps_h = max(len(lines) for lines in cap_lines) * line_h
    note_h = round(note_font.size * 1.6) if note and hasattr(note_font, 'size') else 0
    top = pad + title_h + caps_h + round(10 * unit)
    sheet = Image.new('RGB', (w + 2 * pad, top + h + note_h + pad), BG)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad, pad), title, font=title_font, fill=INK)
    y0 = pad + title_h
    for cx, lines in zip(centers, cap_lines):
        for i, line in enumerate(lines):
            lw = draw.textlength(line, font=cap_font)
            draw.text((pad + cx - lw / 2, y0 + i * line_h), line, font=cap_font, fill=INK)
    sheet.paste(body, (pad, top))
    if note:
        draw.text((pad, top + h + round(8 * unit)), note, font=note_font, fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    print(f'{out.name}: {sheet.size[0]}x{sheet.size[1]}px, {out.stat().st_size / 1024:.0f} KB')


def optimise_png(src: Path, out: Path) -> None:
    with Image.open(src) as im:
        rgb = im.convert('RGB')
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out, optimize=True)
    print(f'{out.name}: {rgb.size[0]}x{rgb.size[1]}px, {out.stat().st_size / 1024:.0f} KB')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gif')
    g.add_argument('--frames', required=True, type=Path)
    g.add_argument('--out', required=True, type=Path)
    g.add_argument('--fps', type=float, default=20)
    g.add_argument('--scale', type=float, default=1.0)
    g.add_argument('--max-bytes', type=int, default=2_500_000)
    g.add_argument('--keep', default=KEEP_DEFAULT, help='comma-separated #rrggbb colours reserved in the palette')
    g.add_argument('--require', default='', help='comma-separated #rrggbb colours that must appear in the GIF')
    s = sub.add_parser('sheet')
    s.add_argument('--strip', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    s.add_argument('--centers', required=True, help='comma-separated x of each column, in strip pixels')
    s.add_argument('--captions', required=True, help='"|"-separated, one per column')
    s.add_argument('--title', default='')
    s.add_argument('--note', default='')
    p = sub.add_parser('png')
    p.add_argument('--src', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == 'gif':
        make_gif(a.frames, a.out, a.fps, a.scale, a.max_bytes, _colours(a.keep), _colours(a.require))
    elif a.cmd == 'sheet':
        centers = [float(x) for x in a.centers.split(',')]
        captions = a.captions.split('|')
        if len(centers) != len(captions):
            raise SystemExit('--centers and --captions need the same number of items')
        make_sheet(a.strip, a.out, centers, captions, a.title, a.note)
    else:
        optimise_png(a.src, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
