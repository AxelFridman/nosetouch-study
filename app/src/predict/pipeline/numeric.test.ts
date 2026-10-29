/**
 * The numpy / scipy primitives against values computed by numpy 2.4.6 / scipy 1.17.1 on synthetic inputs
 * (testdata/numpy_reference.json, written by testdata/make_numpy_reference.py). Bitwise unless stated.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { linspace, norm3, norm3Dot, npInterp, searchLeft, searchRight } from './numeric';
import { fillGapsLinear, finiteRuns, lowpassColumn } from './preprocess';

type Num = number | null;
const ref = JSON.parse(readFileSync(new URL('./testdata/numpy_reference.json', import.meta.url), 'utf8')) as {
  interp: { xp: number[]; fp: number[]; x: Num[]; y: Num[] };
  interp1: { xp: number[]; fp: number[]; x: number[]; y: number[] };
  linspace: { a: number; b: number; n: number; y: number[] }[];
  lowpass: { x: number[]; y: number[] }[];
  lowpass_runs: { x: Num[]; y: Num[] };
  norm_dot: { v: number[][]; n: number[] };
  fill: { t: number[]; x: Num[][]; y: Num[][] };
};
const nan = (v: Num): number => (v === null ? NaN : v);
const same = (a: number, b: number): boolean => Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b));

describe('numpy primitives', () => {
  it('searchsorted left and right', () => {
    const a = [1, 2, 2, 2, 5];
    expect([0, 1, 2, 3, 5, 6].map((v) => searchLeft(a, v))).toEqual([0, 0, 1, 4, 4, 5]);
    expect([0, 1, 2, 3, 5, 6].map((v) => searchRight(a, v))).toEqual([0, 1, 4, 4, 5, 5]);
  });

  it('np.interp, including the edges, an exact last point and NaN', () => {
    const { xp, fp, x, y } = ref.interp;
    x.forEach((xi, i) => expect(same(npInterp(nan(xi), xp, fp), nan(y[i])), `x=${xi}`).toBe(true));
    ref.interp1.x.forEach((xi, i) => expect(npInterp(xi, ref.interp1.xp, ref.interp1.fp)).toBe(ref.interp1.y[i]));
    expect(() => npInterp(1, [], [])).toThrow(RangeError);
  });

  it('np.linspace', () => {
    for (const c of ref.linspace) {
      const y = linspace(c.a, c.b, c.n);
      expect(Array.from(y)).toEqual(c.y);
      y.forEach((v, i) => expect(Object.is(v, c.y[i])).toBe(true));
    }
  });

  it('np.linalg.norm of one vector (BLAS ddot with fused multiply-adds) and of rows (left to right)', () => {
    const { v, n } = ref.norm_dot;
    let plainMismatch = 0;
    v.forEach((r, i) => {
      expect(norm3Dot(r[0], r[1], r[2])).toBe(n[i]);
      if (norm3(r[0], r[1], r[2]) !== n[i]) plainMismatch++;
    });
    // the plain left-to-right sum is the row-norm order and differs from BLAS now and then (not a requirement)
    expect(plainMismatch).toBeLessThan(v.length);
  });
});

describe('scipy sosfiltfilt as motionlab.preprocess.lowpass uses it', () => {
  it('single runs of 3, 4, 9, 16, 17, 40 and 100 samples (pad length min(15, n - 1)) are bit for bit scipy', () => {
    for (const c of ref.lowpass) {
      const col = Float64Array.from(c.x);
      lowpassColumn(col, col.length);
      col.forEach((v, i) => expect(Object.is(v, c.y[i]), `n=${c.x.length} i=${i}`).toBe(true));
    }
  });

  it('runs separated by NaN are filtered separately; a run shorter than 3 is copied', () => {
    const col = Float64Array.from(ref.lowpass_runs.x.map(nan));
    expect(finiteRuns(col, col.length)).toEqual([
      [0, 2],
      [3, 8],
      [10, 30],
    ]);
    lowpassColumn(col, col.length);
    col.forEach((v, i) => expect(same(v, nan(ref.lowpass_runs.y[i])), `i=${i}`).toBe(true));
  });
});

describe('fill_gaps_linear', () => {
  it('interpolates NaN rows per column and holds the edges', () => {
    const { t, x, y } = ref.fill;
    const data = new Float64Array(3 * t.length);
    x.forEach((r, i) => r.forEach((v, c) => (data[3 * i + c] = nan(v))));
    const out = fillGapsLinear(t, data, t.length);
    y.forEach((r, i) => r.forEach((v, c) => expect(same(out[3 * i + c], nan(v)), `row ${i} col ${c}`).toBe(true)));
  });
});
