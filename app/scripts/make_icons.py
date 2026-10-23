"""Draws the NoseTouch app icons into app/public/icons/ (Pillow; run it again after a design change).

Artwork: a dark-blue square with a white phone outline and a motion arc that ends in a dot (the nose
tip) above the phone. It is defined ONCE, in 512-unit coordinates, and rendered as
  icon-192.png, icon-512.png  purpose 'any': rounded square with transparent corners
  maskable-512.png            purpose 'maskable': full-bleed square, artwork shrunk into the safe zone
                              (Android crops maskable icons to a circle or squircle; only the centred
                              circle of radius 40 % is guaranteed to stay visible)
  apple-touch-icon.png        180x180, full-bleed and opaque: iOS rounds the corners itself and would
                              render transparent pixels black
  icon.svg                    the favicon: same geometry as icon-512.png
PNGs are drawn 4x supersampled and downscaled (anti-aliasing). The output is deterministic, so
re-running without a design change leaves the committed files byte-identical.

Usage (from app/):  py -3 scripts/make_icons.py [--out public/icons]   (npm run icons)
Needs Pillow (pip install pillow). Only the generated files are committed, not a Pillow dependency.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

UNITS = 512  # artwork coordinate system
SUPERSAMPLE = 4
CORNER_RADIUS = 112  # rounded square of the 'any' icons (~22 %, close to the iOS/Android look)
GRADIENT_TOP = '#2747c4'
GRADIENT_BOTTOM = '#14246a'
INK = '#ffffff'
MASKABLE_SCALE = 0.8  # farthest artwork point ~207 units from the centre -> ~166, inside the 204.8 safe circle
APPLE_SCALE = 0.92

# Strokes are given by their centre line (as in SVG); the PNG renderer converts them for Pillow.
PHONE = dict(x=256, y=186, w=128, h=232, r=26, sw=22)  # rounded-rectangle outline
SPEAKER = dict(x=300, y=208, w=40, h=10)  # pill at the top of the screen
ARC = dict(cx=320, cy=320, r=210, a0=196, a1=270, sw=18)  # degrees, clockwise from 3 o'clock (y down)
NOSE = dict(cx=320, cy=110, r=24)  # = the arc's end point (cx, cy - r)


def _hex(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _scaled(scale: float):
    """Coordinate transform: shrink the artwork about the centre by `scale`."""
    c = UNITS / 2

    def pt(x: float, y: float) -> tuple[float, float]:
        return c + (x - c) * scale, c + (y - c) * scale

    return pt


def _arc_point(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    t = math.radians(deg)
    return cx + r * math.cos(t), cy + r * math.sin(t)


# ───────────────────────────── PNG ─────────────────────────────


def _gradient(n: int) -> Image.Image:
    ramp = Image.linear_gradient('L').resize((n, n), Image.BICUBIC)  # 0 at the top, 255 at the bottom
    top = Image.new('RGBA', (n, n), _hex(GRADIENT_TOP) + (255,))
    bottom = Image.new('RGBA', (n, n), _hex(GRADIENT_BOTTOM) + (255,))
    return Image.composite(bottom, top, ramp)


def render_png(size: int, *, rounded: bool, scale: float) -> Image.Image:
    n = size * SUPERSAMPLE
    k = n / UNITS  # artwork units -> supersampled pixels
    pt = _scaled(scale)
    img = _gradient(n)
    d = ImageDraw.Draw(img)
    ink = _hex(INK) + (255,)

    def disc(cx: float, cy: float, r: float) -> None:
        x, y = pt(cx, cy)
        rr = r * scale
        d.ellipse(((x - rr) * k, (y - rr) * k, (x + rr) * k, (y + rr) * k), fill=ink)

    # phone: Pillow draws an outline INSIDE its box, so grow the box by half the stroke
    p = PHONE
    x0, y0 = pt(p['x'] - p['sw'] / 2, p['y'] - p['sw'] / 2)
    x1, y1 = pt(p['x'] + p['w'] + p['sw'] / 2, p['y'] + p['h'] + p['sw'] / 2)
    d.rounded_rectangle(
        (x0 * k, y0 * k, x1 * k, y1 * k),
        radius=(p['r'] + p['sw'] / 2) * scale * k,
        outline=ink,
        width=round(p['sw'] * scale * k),
    )
    s = SPEAKER
    x0, y0 = pt(s['x'], s['y'])
    x1, y1 = pt(s['x'] + s['w'], s['y'] + s['h'])
    d.rounded_rectangle((x0 * k, y0 * k, x1 * k, y1 * k), radius=s['h'] / 2 * scale * k, fill=ink)

    # motion arc: Pillow's arc band also grows inwards from its box; round caps are discs
    a = ARC
    cx, cy = pt(a['cx'], a['cy'])
    outer = (a['r'] + a['sw'] / 2) * scale
    d.arc(
        ((cx - outer) * k, (cy - outer) * k, (cx + outer) * k, (cy + outer) * k),
        a['a0'],
        a['a1'],
        fill=ink,
        width=round(a['sw'] * scale * k),
    )
    for deg in (a['a0'], a['a1']):
        disc(*_arc_point(a['cx'], a['cy'], a['r'], deg), a['sw'] / 2)
    disc(NOSE['cx'], NOSE['cy'], NOSE['r'])

    if rounded:
        mask = Image.new('L', (n, n), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=CORNER_RADIUS * k, fill=255)
        img.putalpha(mask)
    # Pillow resamples RGBA premultiplied, so the transparent corners get no dark fringe
    return img.resize((size, size), Image.LANCZOS)


# ───────────────────────────── SVG ─────────────────────────────


def _num(v: float) -> str:
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return '0' if s == '-0' else s


def render_svg() -> str:
    p, s, a, nose = PHONE, SPEAKER, ARC, NOSE
    x0, y0 = _arc_point(a['cx'], a['cy'], a['r'], a['a0'])
    x1, y1 = _arc_point(a['cx'], a['cy'], a['r'], a['a1'])
    large = 1 if (a['a1'] - a['a0']) % 360 > 180 else 0
    arc_d = f"M{_num(x0)} {_num(y0)}A{_num(a['r'])} {_num(a['r'])} 0 {large} 1 {_num(x1)} {_num(y1)}"
    return '\n'.join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {UNITS} {UNITS}">',
            '  <defs>',
            '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">',
            f'      <stop offset="0" stop-color="{GRADIENT_TOP}"/>',
            f'      <stop offset="1" stop-color="{GRADIENT_BOTTOM}"/>',
            '    </linearGradient>',
            '  </defs>',
            f'  <rect width="{UNITS}" height="{UNITS}" rx="{CORNER_RADIUS}" fill="url(#bg)"/>',
            f'  <g fill="none" stroke="{INK}" stroke-linecap="round">',
            f"    <rect x=\"{p['x']}\" y=\"{p['y']}\" width=\"{p['w']}\" height=\"{p['h']}\" rx=\"{p['r']}\""
            f" stroke-width=\"{p['sw']}\"/>",
            f"    <path d=\"{arc_d}\" stroke-width=\"{a['sw']}\"/>",
            '  </g>',
            f'  <g fill="{INK}">',
            f"    <rect x=\"{s['x']}\" y=\"{s['y']}\" width=\"{s['w']}\" height=\"{s['h']}\" rx=\"{_num(s['h'] / 2)}\"/>",
            f"    <circle cx=\"{nose['cx']}\" cy=\"{nose['cy']}\" r=\"{nose['r']}\"/>",
            '  </g>',
            '</svg>',
            '',
        ]
    )


# ───────────────────────────── main ─────────────────────────────


def main(argv: list[str] | None = None) -> int:
    default_out = Path(__file__).resolve().parent.parent / 'public' / 'icons'
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    ap.add_argument('--out', type=Path, default=default_out, help=f'output folder (default: {default_out})')
    args = ap.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    pngs = {
        'icon-192.png': render_png(192, rounded=True, scale=1.0),
        'icon-512.png': render_png(512, rounded=True, scale=1.0),
        'maskable-512.png': render_png(512, rounded=False, scale=MASKABLE_SCALE).convert('RGB'),
        'apple-touch-icon.png': render_png(180, rounded=False, scale=APPLE_SCALE).convert('RGB'),
    }
    for name, img in pngs.items():
        img.save(out / name, 'PNG', optimize=True)
    (out / 'icon.svg').write_text(render_svg(), encoding='utf-8', newline='\n')
    for name in [*pngs, 'icon.svg']:
        f = out / name
        print(f'{f}  {f.stat().st_size:,} bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
