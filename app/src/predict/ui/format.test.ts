import { describe, expect, it } from 'vitest';
import { barLabel, barRightWidth, formatPercent, leftPercent, likelyHand, NBSP, percentOf, roundHalfEven } from './format';

describe('roundHalfEven = Python round()', () => {
  it('rounds halves to the even neighbour, everything else to the nearest', () => {
    expect([0.5, 1.5, 2.5, 3.5, 98.5, 99.5].map(roundHalfEven)).toEqual([0, 2, 2, 4, 98, 100]);
    expect([0.49, 0.51, 86.6, 87.4].map(roundHalfEven)).toEqual([0, 1, 87, 87]);
    expect(roundHalfEven(-0.5)).toBe(0); // Python: round(-0.5) == 0 (an int, no negative zero)
    expect(roundHalfEven(-1.5)).toBe(-2);
  });

  it('percentOf is int(round(100 * p)) on the float64 product, like the Python reference', () => {
    // 100 * 0.995 === 99.5 in float64, and round(99.5) = 100 in Python
    expect(100 * 0.995).toBe(99.5);
    expect(percentOf(0.995)).toBe(100);
    expect(percentOf(0.005)).toBe(0); // 0.5 -> 0 (even)
    expect(percentOf(0.015)).toBe(2); // 1.5 -> 2 (even)
    expect(percentOf(0.865)).toBe(86); // 86.5 -> 86 (even)
    expect(percentOf(0.87)).toBe(87);
    expect(percentOf(0.5)).toBe(50);
  });
});

describe('formatPercent', () => {
  it('never shows 100 % or 0 %', () => {
    expect(formatPercent(100)).toBe(`>99${NBSP}%`);
    expect(formatPercent(0)).toBe(`<1${NBSP}%`);
    expect(formatPercent(percentOf(0.999))).toBe(`>99${NBSP}%`);
    expect(formatPercent(percentOf(0.001))).toBe(`<1${NBSP}%`);
    expect(formatPercent(percentOf(0.994))).toBe(`99${NBSP}%`);
    expect(formatPercent(percentOf(0.006))).toBe(`1${NBSP}%`);
  });

  it('formats an integer percent with a no-break space before the sign', () => {
    expect(formatPercent(87)).toBe('87 %');
    expect(formatPercent(50)).toBe('50 %');
  });

  it('shows a dash when the value is unknown', () => {
    expect(formatPercent(Number.NaN)).toBe('–');
    expect(formatPercent(null)).toBe('–');
    expect(formatPercent(undefined)).toBe('–');
  });
});

describe('the hand bar', () => {
  it('splits at p: the right part is p, the left part 1 - p', () => {
    expect(barRightWidth(0.87)).toBeCloseTo(87, 12);
    expect(barRightWidth(1.2)).toBe(100);
    expect(barRightWidth(-1)).toBe(0);
    expect(barRightWidth(Number.NaN)).toBe(50);
  });

  it('labels both sides, the left share being 100 - percentRight, with the same clamps', () => {
    expect(leftPercent(87)).toBe(13);
    expect(barLabel(87, 'LEFT', 'RIGHT')).toBe(`LEFT 13${NBSP}%, RIGHT 87${NBSP}%`);
    expect(barLabel(100, 'LEFT', 'RIGHT')).toBe(`LEFT <1${NBSP}%, RIGHT >99${NBSP}%`);
    expect(barLabel(0, 'LEFT', 'RIGHT')).toBe(`LEFT >99${NBSP}%, RIGHT <1${NBSP}%`);
  });

  it('names the more likely hand (none at exactly one half)', () => {
    expect(likelyHand(0.87)).toBe('right');
    expect(likelyHand(0.13)).toBe('left');
    expect(likelyHand(0.5)).toBeNull();
    expect(likelyHand(Number.NaN)).toBeNull();
  });
});
