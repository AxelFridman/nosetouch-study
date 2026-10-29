/**
 * numpy / scipy primitives the detector needs, ported so that the TypeScript reproduces the Python reference on
 * the same float64 input (docs/stage3/PORT_SPEC.md section 5.1). Pure functions, float64 throughout.
 *
 * Exactness notes:
 *  - npSum is numpy's pairwise summation of a contiguous 1-D array (numpy loops_utils pairwise_sum, blocks of 8,
 *    started from the identity 0): np.sum / np.mean of a 1-D array are bitwise reproducible with it.
 *  - a reduction over the rows of an (n, 3) array along axis 0 (rest means, trapezoid) is a plain left-to-right
 *    sum in numpy: use seqSum-style loops for those.
 *  - npInterp is numpy's arr_interp (compiled_base.c), including its exact-knot and NaN-slope rules.
 *  - sosfiltInPlace is scipy's _sosfilt direct-form-II-transposed recursion, same evaluation order.
 *  - library functions (exp, log, sin, cos, atan2, sqrt is exact) may differ by 1 ulp between the C runtime and V8.
 */
import type { Rows3 } from '../types';

/** Python's round() of a float to an integer: half to even (int(round(x)); PORT_SPEC finding 9). */
export function roundHalfEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const f = Math.floor(x);
  const diff = x - f; // exact for |x| < 2^52
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** numpy's pairwise sum of a[start .. start + n - 1] (np.sum of a contiguous 1-D float64 array). */
export function npSum(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[start + i];
    return res;
  }
  if (n <= 128) {
    const r0 = [a[start], a[start + 1], a[start + 2], a[start + 3], a[start + 4], a[start + 5], a[start + 6], a[start + 7]];
    let i = 8;
    const stop = n - (n % 8);
    for (; i < stop; i += 8) {
      for (let j = 0; j < 8; j++) r0[j] += a[start + i + j];
    }
    // numpy: res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    let res = ((r0[0] + r0[1]) + (r0[2] + r0[3])) + ((r0[4] + r0[5]) + (r0[6] + r0[7]));
    for (; i < n; i++) res += a[start + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return npSum(a, start, n2) + npSum(a, start + n2, n - n2);
}

/** np.mean of a contiguous 1-D float64 array (pairwise sum / n). */
export function npMean(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  return npSum(a, start, n) / n;
}

/** np.max of a 1-D array: NaN as soon as any value is NaN (numpy propagates NaN). Requires n >= 1. */
export function npMax(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  let m = a[start];
  for (let i = 0; i < n; i++) {
    const v = a[start + i];
    if (Number.isNaN(v)) return NaN;
    if (v > m) m = v;
  }
  return m;
}

/** np.min of a 1-D array (NaN propagates). Requires n >= 1. */
export function npMin(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  let m = a[start];
  for (let i = 0; i < n; i++) {
    const v = a[start + i];
    if (Number.isNaN(v)) return NaN;
    if (v < m) m = v;
  }
  return m;
}

/** np.argmax (offset from start): the first maximum; the first NaN when there is one (numpy treats NaN as max). */
export function npArgmax(a: ArrayLike<number>, start = 0, n = a.length - start): number {
  let best = 0;
  let m = a[start];
  if (Number.isNaN(m)) return 0;
  for (let i = 1; i < n; i++) {
    const v = a[start + i];
    if (Number.isNaN(v)) return i;
    if (v > m) {
      m = v;
      best = i;
    }
  }
  return best;
}

/** Python's max(a, b) on floats: b only when b > a (so max(0.0, nan) is 0.0, unlike Math.max). */
export function pyMax(a: number, b: number): number {
  return b > a ? b : a;
}

/** Python's min(a, b) on floats: b only when b < a. */
export function pyMin(a: number, b: number): number {
  return b < a ? b : a;
}

/**
 * np.median of the values (a copy is sorted; the input is untouched). NaN when any value is NaN or the input is
 * empty; an even count averages the two middle values as numpy does ((a + b) / 2).
 */
export function npMedian(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) return NaN;
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (Number.isNaN(v)) return NaN;
    s[i] = v;
  }
  s.sort();
  const h = n >> 1;
  return n % 2 === 1 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** np.searchsorted(a, v, side='left'): the first index i with a[i] >= v (a sorted ascending, no NaN). */
export function searchsortedLeft(a: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** np.searchsorted(a, v, side='right'): the first index i with a[i] > v. */
export function searchsortedRight(a: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * np.interp(x, xp, fp) for one x (numpy compiled_base.c arr_interp): NaN x gives NaN; x < xp[0] gives fp[0];
 * x > xp[last] gives fp[last]; x == xp[last] gives fp[last]; x == xp[j] gives fp[j]; otherwise
 * slope * (x - xp[j]) + fp[j] with slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]) and numpy's NaN retries.
 * xp ascending without NaN; the length is xp.length (fp may be longer: only its first xp.length values are read).
 */
export function npInterp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>, len = xp.length): number {
  if (Number.isNaN(x)) return x;
  if (len === 1) return fp[0]; // left value, right value and the knot are all fp[0]
  if (x < xp[0]) return fp[0];
  if (x > xp[len - 1]) return fp[len - 1];
  // j with xp[j] <= x < xp[j+1], or len - 1 when x == xp[len - 1]
  let lo = 0;
  let hi = len; // invariant: xp[lo] <= x, and x < xp[hi] (hi = len means +inf)
  while (hi - lo > 1) {
    const mid = (lo + hi) >>> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const j = lo;
  if (j === len - 1) return fp[j];
  if (xp[j] === x) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  let r = slope * (x - xp[j]) + fp[j];
  if (Number.isNaN(r)) {
    r = slope * (x - xp[j + 1]) + fp[j + 1];
    if (Number.isNaN(r) && fp[j] === fp[j + 1]) r = fp[j];
  }
  return r;
}

/**
 * scipy.signal.sosfilt over x in place, state z (n_sections * 2, mutated: the final state on return).
 * Direct-form II transposed, scipy _sosfilt.pyx order: y = b0*x + z0; z0 = (b1*x - a1*y) + z1; z1 = b2*x - a2*y.
 */
export function sosfiltInPlace(sos: readonly (readonly number[])[], x: Float64Array, z: Float64Array): void {
  const ns = sos.length;
  for (let i = 0; i < x.length; i++) {
    let xc = x[i];
    for (let s = 0; s < ns; s++) {
      const c = sos[s];
      const y = c[0] * xc + z[2 * s];
      z[2 * s] = c[1] * xc - c[4] * y + z[2 * s + 1];
      z[2 * s + 1] = c[2] * xc - c[5] * y;
      xc = y;
    }
    x[i] = xc;
  }
}

/** One sample through sosfilt (same recursion as sosfiltInPlace); z is updated in place. */
export function sosfiltStep(sos: readonly (readonly number[])[], x: number, z: Float64Array): number {
  let xc = x;
  for (let s = 0; s < sos.length; s++) {
    const c = sos[s];
    const y = c[0] * xc + z[2 * s];
    z[2 * s] = c[1] * xc - c[4] * y + z[2 * s + 1];
    z[2 * s + 1] = c[2] * xc - c[5] * y;
    xc = y;
  }
  return xc;
}

/** The initial state zi * x0 of every section (scipy: sosfilt_zi(sos) * x[0]). */
export function scaledState(zi: readonly (readonly number[])[], x0: number): Float64Array {
  const z = new Float64Array(zi.length * 2);
  for (let s = 0; s < zi.length; s++) {
    z[2 * s] = zi[s][0] * x0;
    z[2 * s + 1] = zi[s][1] * x0;
  }
  return z;
}

/**
 * motionlab.preprocess.fill_gaps_linear of ONE column: np.interp over the NaN rows from the finite rows (edges
 * hold the nearest valid value). A column without NaN, or with no finite value, is returned unchanged (copied).
 */
export function fillGapsColumn(t: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const n = x.length;
  const out = Float64Array.from(x);
  let nOk = 0;
  for (let i = 0; i < n; i++) if (Number.isFinite(out[i])) nOk++;
  if (nOk === n || nOk === 0) return out;
  const tv = new Float64Array(nOk);
  const xv = new Float64Array(nOk);
  for (let i = 0, k = 0; i < n; i++) {
    if (Number.isFinite(out[i])) {
      tv[k] = t[i];
      xv[k] = out[i];
      k++;
    }
  }
  for (let i = 0; i < n; i++) if (!Number.isFinite(out[i])) out[i] = npInterp(t[i], tv, xv);
  return out;
}

/** fill_gaps_linear over the three columns of an (n, 3) array (each column on its own, like the Python loop). */
export function fillGapsRows(t: ArrayLike<number>, X: Rows3): Rows3 {
  const n = X.n;
  const out = new Float64Array(n * 3);
  for (let j = 0; j < 3; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = X.data[3 * i + j];
    const filled = fillGapsColumn(t, col);
    for (let i = 0; i < n; i++) out[3 * i + j] = filled[i];
  }
  return { n, data: out };
}

/** |v| of a 3-vector as np.linalg.norm(axis=-1) computes it: sqrt((x*x + y*y) + z*z). */
export function norm3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

/** sigmoid with m4_models.sigmoid's clip of the argument to [-40, 40]. */
export function sigmoidClipped(z: number, clip = 40): number {
  const c = z < -clip ? -clip : z > clip ? clip : z; // np.clip keeps NaN
  return 1.0 / (1.0 + Math.exp(-c));
}
