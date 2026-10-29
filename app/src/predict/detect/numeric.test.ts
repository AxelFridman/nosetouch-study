/** numpy / scipy primitives of detect/ against values computed by numpy 2.4 (analysis/.venv) on the same input. */
import { describe, expect, it } from 'vitest';
import {
  fillGapsColumn,
  npArgmax,
  npInterp,
  npMax,
  npMean,
  npMedian,
  npSum,
  pyMax,
  roundHalfEven,
  searchsortedLeft,
  searchsortedRight,
  sosfiltInPlace,
} from './numeric';

const SC = [0.01, 0.1, 1.0, 10.0, 100.0];
/** x_k = (((k * 7919) % 1000 - 500) / 37) * SC[k % 5]: exactly the same doubles in Python and JS */
const arr = (n: number): Float64Array => Float64Array.from({ length: n }, (_, k) => ((((k * 7919) % 1000) - 500) / 37) * SC[k % 5]);

// [np.sum, np.mean, np.median] of arr(n), from numpy 2.4
const NUMPY: Record<number, [number, number, number]> = {
  1: [-0.13513513513513514, -0.13513513513513514, -0.13513513513513514],
  5: [555.2675675675675, 111.05351351351351, 9.135135135135135],
  8: [553.5202702702703, 69.19003378378379, 0.5851351351351352],
  9: [513.5202702702703, 57.05780780780781, 0.03783783783783784],
  17: [991.3324324324325, 58.31367249602544, 0.03783783783783784],
  100: [-1117.4729729729734, -11.174729729729734, -0.031081081081081083],
  128: [-288.0094594594584, -2.250073902027019, 0.002027027027027027],
  129: [-252.3337837837837, -1.956075843285145, 0.005405405405405406],
  300: [-619.9864864864857, -2.0666216216216187, 0.004054054054054054],
  1000: [-834.4594594594578, -0.8344594594594578, -0.0006756756756756757],
};

describe('numeric primitives', () => {
  it('npSum / npMean / npMedian are bitwise numpy (pairwise summation, blocks of 8)', () => {
    for (const [n, [sum, mean, median]] of Object.entries(NUMPY)) {
      const a = arr(Number(n));
      expect(npSum(a), `sum n=${n}`).toBe(sum);
      expect(npMean(a), `mean n=${n}`).toBe(mean);
      expect(npMedian(a), `median n=${n}`).toBe(median);
    }
    // a naive left-to-right sum is NOT numpy's for long arrays
    const a = arr(1000);
    let naive = 0;
    for (const v of a) naive += v;
    expect(naive).not.toBe(NUMPY[1000][0]);
  });

  it('npSum honours start / n (a slice), npMedian handles NaN and even counts', () => {
    const a = arr(300);
    expect(npSum(a, 100, 128)).toBe(npSum(a.slice(100, 228)));
    expect(npMedian([3, 1, 2, 4])).toBe(2.5);
    expect(npMedian([1, NaN, 2])).toBeNaN();
    expect(npMedian([])).toBeNaN();
  });

  it('roundHalfEven is Python round()', () => {
    const xs = [0.5, 1.5, 2.5, -0.5, -1.5, 12.5, 100 * 0.125, 100 * 0.135, 100 * 0.145, 100 * 0.875, 0.49999999999999994, 3.7, -3.7];
    // Python: [0, 2, 2, 0, -2, 12, 12, 14, 14, 88, 0, 4, -4]
    expect(xs.map(roundHalfEven).map((v) => v + 0)).toEqual([0, 2, 2, 0, -2, 12, 12, 14, 14, 88, 0, 4, -4]);
    expect(Math.round(12.5)).toBe(13); // why the helper exists
  });

  it('npInterp follows numpy arr_interp (edges, knots, last knot)', () => {
    const xp = [0.0, 1.0, 2.5, 4.0];
    const fp = [1.0, 3.0, -2.0, 5.0];
    const cases: [number, number][] = [
      [-1.0, 1.0],
      [0.0, 1.0],
      [0.5, 2.0],
      [1.0, 3.0],
      [2.0, -0.3333333333333335],
      [2.5, -2.0],
      [3.9, 4.533333333333333],
      [4.0, 5.0],
      [7.0, 5.0],
    ];
    for (const [x, want] of cases) expect(npInterp(x, xp, fp), `x=${x}`).toBe(want);
    expect(npInterp(NaN, xp, fp)).toBeNaN();
    expect(npInterp(5, [2], [7])).toBe(7);
  });

  it('searchsorted left / right', () => {
    const a = [1, 2, 2, 3];
    expect([searchsortedLeft(a, 2), searchsortedRight(a, 2), searchsortedLeft(a, 0), searchsortedRight(a, 9)]).toEqual([1, 3, 0, 4]);
  });

  it('max / argmax propagate NaN like numpy; pyMax is Python max', () => {
    expect(npMax([1, NaN, 3])).toBeNaN();
    expect(npArgmax([1, 5, 5, 2])).toBe(1);
    expect(npArgmax([1, NaN, 9])).toBe(1);
    expect(pyMax(0.0, NaN)).toBe(0.0);
    expect(Math.max(0.0, NaN)).toBeNaN();
  });

  it('fillGapsColumn interpolates inner gaps and holds the edges (fill_gaps_linear)', () => {
    const t = [0, 10, 20, 30, 40, 50];
    expect(Array.from(fillGapsColumn(t, [NaN, 1, NaN, NaN, 4, NaN]))).toEqual([1, 1, 2, 3, 4, 4]);
    expect(Array.from(fillGapsColumn(t, [NaN, NaN, NaN, NaN, NaN, NaN])).every(Number.isNaN)).toBe(true);
  });

  it('sosfilt: a unit DC input settles at the filter gain with the steady-state initial state', () => {
    // activity filter of the m4 bundle: butter(2, 5 Hz, fs=60) and sosfilt_zi
    const sos = [[0.04948995626867706, 0.09897991253735412, 0.04948995626867706, 1.0, -1.2796324249978088, 0.47759225007251704]];
    const z = new Float64Array([0.9505100437313232, -0.4281022938038401]);
    const x = new Float64Array(50).fill(1);
    sosfiltInPlace(sos, x, z);
    for (const v of x) expect(Math.abs(v - 1)).toBeLessThan(1e-12);
  });
});
