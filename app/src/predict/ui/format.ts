/**
 * Number formatting of the P(right) display (docs/stage3/APP_SPEC.md section 5.1).
 *  - percent = int(round(100 * p)) with Python's round-half-to-even (the engine's Decision.percentRight
 *    and ModelComparison.percentRight already are; roundHalfEven is for values the UI derives itself);
 *  - the display never says "100 %" or "0 %": a rounded 100 shows ">99 %" (p >= 0.995) and a rounded 0
 *    shows "<1 %" (p <= 0.005), both with a no-break space before the sign.
 */
import type { Hand } from '../types';

export const NBSP = ' ';

/** Python's round() of a float to an integer: halves go to the even neighbour. */
export function roundHalfEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const f = Math.floor(x);
  const d = x - f; // exact for |x| < 2^52
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** int(round(100 * p)), the engine's percentRight. */
export function percentOf(p: number): number {
  return roundHalfEven(100 * p);
}

/** "87 %", ">99 %", "<1 %"; "–" when unknown. */
export function formatPercent(percent: number | null | undefined): string {
  if (percent === null || percent === undefined || !Number.isFinite(percent)) return '–';
  if (percent >= 100) return `>99${NBSP}%`;
  if (percent <= 0) return `<1${NBSP}%`;
  return `${Math.round(percent)}${NBSP}%`;
}

/** The left side's share shown under the bar: 100 - percentRight, clamped the same way. */
export function leftPercent(percentRight: number): number {
  return 100 - percentRight;
}

/** The more likely hand (exactly 50 %: null). */
export function likelyHand(p: number): Hand | null {
  if (!Number.isFinite(p) || p === 0.5) return null;
  return p > 0.5 ? 'right' : 'left';
}

/** Width of the RIGHT part of the bar in percent of the bar (0..100), from the probability itself. */
export function barRightWidth(p: number): number {
  if (!Number.isFinite(p)) return 50;
  return Math.min(100, Math.max(0, 100 * p));
}

/** "LEFT 13 %, RIGHT 87 %" (the bar's accessible label). */
export function barLabel(percentRight: number, leftWord: string, rightWord: string): string {
  return `${leftWord} ${formatPercent(leftPercent(percentRight))}, ${rightWord} ${formatPercent(percentRight)}`;
}

/** seconds with one decimal (debug panel) */
export function seconds(ms: number | null | undefined): string {
  return ms === null || ms === undefined || !Number.isFinite(ms) ? '–' : (ms / 1000).toFixed(1);
}

/** a probability with two decimals (debug panel) */
export function prob(p: number | null | undefined): string {
  return p === null || p === undefined || !Number.isFinite(p) ? '–' : p.toFixed(2);
}
