/**
 * Dependency-free <canvas> line plot (trial review, researcher sandbox).
 *
 *  - DPR-aware (crisp on phones); sizes itself to its container width and redraws whenever its props
 *    change (pass new arrays to update; identical arrays are not redrawn).
 *  - Non-finite values (NaN, ±Infinity) break the line instead of being drawn as 0; empty data shows
 *    `emptyLabel`.
 *  - Time always runs left→right, also on RTL pages (the canvas box is dir="ltr"); the HTML legend
 *    follows the page direction.
 *  - Colors may be any CSS color or 'var(--token)' / 'var(--token, fallback)', resolved against the
 *    canvas so light/dark themes work.
 *  - x tick labels are seconds (t is in ms); the y range is [min(0, data), nice(max)] unless fixed.
 *
 * Also exports pure helpers (no DOM) used by the screens and unit tests: niceTicks/niceCeil,
 * ema, motionMagnitudes (the same |ω| / |a| signals the online segmenter uses) and drawSignalPlot.
 */
import { useEffect, useRef, useState } from 'preact/hooks';
import type { MotionStream } from '../core/types';

export interface SignalSeries {
  values: number[];
  color: string;
  label: string;
}

export interface SignalMarker {
  t: number;
  label: string;
  color?: string;
}

export interface SignalHLine {
  y: number;
  label?: string;
  color?: string;
}

export interface SignalPlotProps {
  t: number[];
  series: SignalSeries[];
  markers?: SignalMarker[];
  /** CSS px (default 160) */
  height?: number;
  /** fixed top of the y axis (default: a nice value above the data) */
  yMax?: number;
  /** fixed bottom of the y axis (default: min(0, data min)) */
  yMin?: number;
  /** fixed time window in ms (default: first..last finite t) */
  xRange?: [number, number];
  /** horizontal reference lines (thresholds); they are kept inside the y range */
  hLines?: SignalHLine[];
  /** draw y tick labels (default true; false for normalized plots) */
  yTicks?: boolean;
  ariaLabel?: string;
  emptyLabel?: string;
  testId?: string;
}

// ───────────────────────────── pure helpers ─────────────────────────────

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

// Axis limits use a fine sequence (gravity 9.81 → 12, not 20); tick steps a coarse one (round labels).
const FINE_STEPS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
const COARSE_STEPS = [1, 2, 2.5, 5, 10];

function niceFrom(v: number, steps: number[]): number {
  if (!isNum(v) || v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  const step = steps.find((s) => norm <= s + 1e-9) ?? 10;
  return Number((step * mag).toPrecision(12));
}

/** Smallest "nice" axis limit (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 × 10^k) ≥ v; 1 for v ≤ 0 or non-finite. */
export function niceCeil(v: number): number {
  return niceFrom(v, FINE_STEPS);
}

/** Smallest "nice" tick step (1, 2, 2.5, 5 × 10^k) ≥ v; 1 for v ≤ 0 or non-finite. */
export function niceStep(v: number): number {
  return niceFrom(v, COARSE_STEPS);
}

/** Decimals needed to print multiples of `step` exactly (0..4). */
export function stepDecimals(step: number): number {
  let d = 0;
  while (d < 4 && Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) > 1e-6) d++;
  return d;
}

/** Evenly spaced "nice" tick values inside [min, max] (about `target` of them). */
export function niceTicks(min: number, max: number, target = 4): number[] {
  if (!isNum(min) || !isNum(max) || max <= min) return [];
  const step = niceStep((max - min) / Math.max(1, target));
  const d = stepDecimals(step);
  const first = Math.ceil(min / step - 1e-9);
  const out: number[] = [];
  for (let i = first; out.length < 50; i++) {
    const v = Number((i * step).toFixed(d));
    if (v > max + step * 1e-9) break;
    out.push(v === 0 ? 0 : v); // no "-0"
  }
  return out;
}

/** EMA(x; τ) with S_0 = x_0 and k_i = 1 − exp(−dt_i/τ) (τ ≤ 0 → no smoothing), as in online_v1. */
export function ema(t: number[], x: number[], tauMs: number): number[] {
  const n = Math.min(t.length, x.length);
  const out = new Array<number>(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    if (i === 0 || !(tauMs > 0)) s = x[i];
    else s += (1 - Math.exp(-Math.max(0, t[i] - t[i - 1]) / tauMs)) * (x[i] - s);
    out[i] = s;
  }
  return out;
}

export type AccSource = 'linear' | 'highpass' | 'none';

export interface MotionMagnitudes {
  t: number[];
  /** smoothed |ω| (deg/s) */
  gyro: number[];
  /** smoothed |a| without gravity (m/s²) */
  acc: number[];
  accSource: AccSource;
}

function vecAt(x: (number | null)[], y: (number | null)[], z: (number | null)[], i: number): [number, number, number] | null {
  const a = x[i];
  const b = y[i];
  const c = z[i];
  return isNum(a) && isNum(b) && isNum(c) ? [a, b, c] : null;
}

/**
 * Smoothed |ω| and |a_lin| of a recorded motion stream, following the online_v1 signal rules:
 * the acceleration source is decided by the FIRST sample (acc → 'linear', else accg → 'highpass'
 * with a gravity EMA of gravityTauMs, else 'none'); a missing reading reuses the previous raw value.
 */
export function motionMagnitudes(m: MotionStream, opts: { smoothingMs: number; gravityTauMs: number }): MotionMagnitudes {
  const n = m.t.length;
  const accSource: AccSource =
    n === 0
      ? 'none'
      : vecAt(m.acc_x, m.acc_y, m.acc_z, 0)
        ? 'linear'
        : vecAt(m.accg_x, m.accg_y, m.accg_z, 0)
          ? 'highpass'
          : 'none';
  const gyroRaw = new Array<number>(n);
  const accRaw = new Array<number>(n);
  let g = 0;
  let a = 0;
  let grav: [number, number, number] | null = null;
  let gravT = 0;
  for (let i = 0; i < n; i++) {
    const rr = vecAt(m.rr_alpha, m.rr_beta, m.rr_gamma, i);
    if (rr) g = Math.sqrt(rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]);
    gyroRaw[i] = g;
    if (accSource === 'linear') {
      const v = vecAt(m.acc_x, m.acc_y, m.acc_z, i);
      if (v) a = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    } else if (accSource === 'highpass') {
      const v = vecAt(m.accg_x, m.accg_y, m.accg_z, i);
      if (v) {
        if (!grav) {
          grav = [v[0], v[1], v[2]];
        } else {
          const k = opts.gravityTauMs > 0 ? 1 - Math.exp(-Math.max(0, m.t[i] - gravT) / opts.gravityTauMs) : 1;
          grav = [grav[0] + k * (v[0] - grav[0]), grav[1] + k * (v[1] - grav[1]), grav[2] + k * (v[2] - grav[2])];
        }
        gravT = m.t[i];
        const d0 = v[0] - grav[0];
        const d1 = v[1] - grav[1];
        const d2 = v[2] - grav[2];
        a = Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
      }
    }
    accRaw[i] = a;
  }
  return {
    t: m.t.slice(0, n),
    gyro: ema(m.t, gyroRaw, opts.smoothingMs),
    acc: ema(m.t, accRaw, opts.smoothingMs),
    accSource,
  };
}

/** Largest finite value of the arrays (null if none). */
export function finiteMax(...arrays: number[][]): number | null {
  let best: number | null = null;
  for (const arr of arrays) for (const v of arr) if (isNum(v) && (best === null || v > best)) best = v;
  return best;
}

export interface PlotTheme {
  text: string;
  grid: string;
  axis: string;
  font: string;
  fallbackColor: string;
}

const DEFAULT_THEME: PlotTheme = {
  text: '#586179',
  grid: '#d8dde8',
  axis: '#8a93a8',
  font: '11px system-ui, sans-serif',
  fallbackColor: '#0f172a',
};

/** Resolved x and y ranges of a plot (always non-degenerate). */
export function plotRanges(props: SignalPlotProps): { x: [number, number]; y: [number, number]; hasData: boolean } {
  let x0 = Infinity;
  let x1 = -Infinity;
  for (const t of props.t) {
    if (!isNum(t)) continue;
    if (t < x0) x0 = t;
    if (t > x1) x1 = t;
  }
  if (props.xRange && isNum(props.xRange[0]) && isNum(props.xRange[1])) [x0, x1] = props.xRange;
  if (!isNum(x0) || !isNum(x1)) [x0, x1] = [0, 1];
  if (x1 <= x0) x1 = x0 + 1;

  let lo = Infinity;
  let hi = -Infinity;
  let hasData = false;
  for (const s of props.series) {
    const n = Math.min(s.values.length, props.t.length);
    for (let i = 0; i < n; i++) {
      const v = s.values[i];
      const t = props.t[i];
      if (!isNum(v) || !isNum(t) || t < x0 || t > x1) continue;
      hasData = true;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  for (const h of props.hLines ?? []) {
    if (!isNum(h.y)) continue;
    if (h.y < lo) lo = h.y;
    if (h.y > hi) hi = h.y;
  }
  const y0 = isNum(props.yMin) ? props.yMin : isNum(lo) && lo < 0 ? -niceCeil(-lo * 1.05) : 0;
  let y1 = isNum(props.yMax) ? props.yMax : isNum(hi) && hi > 0 ? niceCeil(hi * 1.05) : y0 < 0 ? 0 : 1;
  if (y1 <= y0) y1 = y0 + 1;
  return { x: [x0, x1], y: [y0, y1], hasData };
}

/**
 * Draws the plot into a 2D context whose transform maps CSS px (w × h). Pure apart from the context
 * calls, so it can be tested with a fake context. `color` resolves CSS colors such as 'var(--x)'.
 */
export function drawSignalPlot(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  props: SignalPlotProps,
  theme: PlotTheme = DEFAULT_THEME,
  color: (c: string) => string = (c) => c,
): void {
  ctx.clearRect(0, 0, w, h);
  const showY = props.yTicks !== false;
  const padL = showY ? 36 : 6;
  const padR = 8;
  const padT = 16;
  const padB = 18;
  const pw = Math.max(1, w - padL - padR);
  const ph = Math.max(1, h - padT - padB);
  const { x: xr, y: yr, hasData } = plotRanges(props);
  const [x0, x1] = xr;
  const [y0, y1] = yr;
  const X = (t: number) => padL + ((t - x0) / (x1 - x0)) * pw;
  const Y = (v: number) => padT + (1 - (v - y0) / (y1 - y0)) * ph;

  ctx.font = theme.font;
  ctx.lineJoin = 'round';
  ctx.setLineDash([]);

  // grid + y labels
  const yTicks = niceTicks(y0, y1, 4);
  const yDec = yTicks.length > 1 ? stepDecimals(yTicks[1] - yTicks[0]) : 0;
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of yTicks) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = v === 0 ? theme.axis : theme.grid;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
    if (showY) {
      ctx.fillStyle = theme.text;
      ctx.fillText(v.toFixed(yDec), padL - 5, y);
    }
  }

  // x labels (seconds)
  const xTicks = niceTicks(x0 / 1000, x1 / 1000, Math.max(2, Math.floor(pw / 70)));
  const xDec = xTicks.length > 1 ? stepDecimals(xTicks[1] - xTicks[0]) : 0;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = theme.text;
  xTicks.forEach((s, i) => {
    const x = X(s * 1000);
    const label = s.toFixed(xDec) + (i === xTicks.length - 1 ? ' s' : '');
    const half = ctx.measureText(label).width / 2;
    ctx.fillText(label, Math.min(Math.max(x, padL + half), padL + pw - half), padT + ph + 4);
  });

  // reference lines
  ctx.setLineDash([2, 3]);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  for (const hl of props.hLines ?? []) {
    if (!isNum(hl.y) || hl.y < y0 || hl.y > y1) continue;
    const y = Math.round(Y(hl.y)) + 0.5;
    const c = color(hl.color ?? theme.axis);
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.stroke();
    if (hl.label) {
      ctx.fillStyle = c;
      ctx.fillText(hl.label, padL + 3, y - 1);
    }
  }
  ctx.setLineDash([]);

  // series (clipped to the plot area; gaps at non-finite values)
  ctx.save();
  ctx.beginPath();
  ctx.rect(padL, padT, pw, ph);
  ctx.clip();
  ctx.lineWidth = 1.75;
  for (const s of props.series) {
    const n = Math.min(s.values.length, props.t.length);
    ctx.strokeStyle = color(s.color || theme.fallbackColor);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const t = props.t[i];
      const v = s.values[i];
      if (!isNum(t) || !isNum(v)) {
        pen = false;
        continue;
      }
      const x = X(t);
      const y = Y(v);
      if (pen) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
  }
  ctx.restore();

  // event markers
  ctx.lineWidth = 1.25;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  // label rows: above the plot, then just inside its top edge when two markers are close together
  // (e.g. the GO beep and the movement onset 250 ms later)
  const rows: [number, number][][] = [[], []];
  for (const m of props.markers ?? []) {
    if (!isNum(m.t) || m.t < x0 || m.t > x1) continue;
    const x = Math.round(X(m.t)) + 0.5;
    const c = color(m.color ?? theme.axis);
    ctx.strokeStyle = c;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    if (m.label) {
      const half = ctx.measureText(m.label).width / 2;
      const cx = Math.min(Math.max(x, half + 1), w - half - 1);
      const span: [number, number] = [cx - half - 3, cx + half + 3];
      const free = (row: [number, number][]) => row.every(([a, b]) => span[1] < a || span[0] > b);
      const r = free(rows[0]) || !free(rows[1]) ? 0 : 1;
      rows[r].push(span);
      ctx.fillStyle = c;
      ctx.fillText(m.label, cx, r === 0 ? padT - 2 : padT + 12);
    }
  }

  if (!hasData) {
    ctx.fillStyle = theme.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(props.emptyLabel ?? '–', padL + pw / 2, padT + ph / 2);
  }
}

/** 'var(--name)' / 'var(--name, fallback)' → computed value; other colors pass through. */
function resolveCssColor(css: CSSStyleDeclaration, c: string, fallback: string): string {
  const m = /^var\(\s*(--[\w-]+)\s*(?:,\s*(.+))?\)$/.exec(c.trim());
  if (!m) return c;
  return css.getPropertyValue(m[1]).trim() || m[2]?.trim() || fallback;
}

function cssVar(css: CSSStyleDeclaration, name: string, fallback: string): string {
  return css.getPropertyValue(name).trim() || fallback;
}

// ───────────────────────────── component ─────────────────────────────

export function SignalPlot(props: SignalPlotProps) {
  const height = props.height ?? 160;
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const measure = () => setWidth(Math.round(box.getBoundingClientRect().width));
    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(measure);
      ro.observe(box);
      return () => ro.disconnect();
    }
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0) return;
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    const bw = Math.round(width * dpr);
    const bh = Math.round(height * dpr);
    if (canvas.width !== bw) canvas.width = bw;
    if (canvas.height !== bh) canvas.height = bh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(canvas);
    const theme: PlotTheme = {
      text: cssVar(css, '--muted', DEFAULT_THEME.text),
      grid: cssVar(css, '--border', DEFAULT_THEME.grid),
      axis: cssVar(css, '--muted', DEFAULT_THEME.axis),
      font: `11px ${cssVar(css, '--font', 'system-ui, sans-serif')}`,
      fallbackColor: cssVar(css, '--text', DEFAULT_THEME.fallbackColor),
    };
    ctx.direction = 'ltr';
    drawSignalPlot(ctx, width, height, props, theme, (c) => resolveCssColor(css, c, theme.fallbackColor));
  }, [
    width,
    height,
    props.t,
    props.series,
    props.markers,
    props.hLines,
    props.yMax,
    props.yMin,
    props.xRange?.[0],
    props.xRange?.[1],
    props.yTicks,
    props.emptyLabel,
  ]);

  return (
    <figure class="signal-plot" data-testid={props.testId}>
      <div class="signal-plot-box" ref={boxRef} dir="ltr" style={{ height: `${height}px` }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={props.ariaLabel ?? props.series.map((s) => s.label).join(', ')}
          style={{ width: '100%', height: `${height}px` }}
        />
      </div>
      {props.series.length > 0 && (
        <figcaption class="signal-plot-legend">
          {props.series.map((s) => (
            <span class="legend-item" key={s.label}>
              <span class="legend-swatch" style={{ background: s.color }} aria-hidden="true" />
              {s.label}
            </span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}
