/**
 * The numpy / scipy primitives the pipeline needs, ported operation by operation (float64, same order of
 * operations) so that the TypeScript reproduces the Python reference bit for bit wherever the libraries do not
 * call into BLAS or a C math library (docs/stage3/PORT_SPEC.md section 5.1):
 *   - np.searchsorted (left / right), np.interp (numpy/_core/src/multiarray/compiled_base.c arr_interp),
 *     np.linspace (numpy/_core/function_base.py);
 *   - scipy.signal.sosfilt (scipy/signal/_sosfilt.pyx, direct form II transposed) and sosfiltfilt with
 *     odd_ext padding and zi scaled by the first sample (scipy/signal/_signaltools.py);
 *   - a 3-vector norm in the order OpenBLAS ddot uses on this machine (fused multiply-adds), for the
 *     np.linalg.norm(1-D) call in gyro_gravity.
 * Reductions: numpy reduces 3 or 4 contiguous values left to right ((a0 + a1) + a2), and an axis-0 mean of a
 * C-ordered (m, 3) array row by row (both checked bitwise against numpy 2.4, PORT_SPEC 5.1).
 */

/** First index i in [lo, hi) with a[i] >= v (numpy searchsorted side='left' on an ascending array). */
export function searchLeft(a: ArrayLike<number>, v: number, lo = 0, hi = a.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** First index i in [lo, hi) with a[i] > v (numpy searchsorted side='right' on an ascending array). */
export function searchRight(a: ArrayLike<number>, v: number, lo = 0, hi = a.length): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * np.interp(x, xp[0:len], fp[0:len]) for one x, with xp strictly increasing (every caller's case: stream grids,
 * the finite rows of a grid, deduplicated sample times). Same branches as numpy's arr_interp:
 *  - one sample point: fp[0] (also for NaN x);
 *  - NaN x: NaN; x < xp[0]: fp[0]; x > xp[last]: fp[last];
 *  - xp[j] <= x < xp[j + 1]: x == xp[j] gives fp[j] exactly, else slope * (x - xp[j]) + fp[j] with
 *    slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j]); a NaN result retries from the right end, then fp[j] when
 *    fp[j] == fp[j+1].
 * numpy raises on an empty xp; so does this.
 */
export function npInterp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>, len = xp.length): number {
  if (len === 1) {
    // compiled_base.c: lenxp == 1 -> (x < xp) ? lval : (x > xp) ? rval : fp  (lval = rval = fp[0])
    return fp[0];
  }
  if (len < 1) throw new RangeError('np.interp: array of sample points is empty');
  if (x !== x) return x; // NaN
  if (x > xp[len - 1]) return fp[len - 1];
  if (x < xp[0]) return fp[0];
  const j = searchRight(xp, x, 0, len) - 1; // xp[j] <= x < xp[j + 1], or len - 1 when x == xp[last]
  if (j === len - 1) return fp[j];
  if (xp[j] === x) return fp[j];
  return interpBetween(x, xp[j], xp[j + 1], fp[j], fp[j + 1]);
}

/**
 * numpy's interpolation step between two sample points (x0 < x < x1): slope * (x - x0) + y0 with
 * slope = (y1 - y0) / (x1 - x0); a NaN result retries from the right end, then y0 when y0 == y1.
 */
export function interpBetween(x: number, x0: number, x1: number, y0: number, y1: number): number {
  const slope = (y1 - y0) / (x1 - x0);
  let r = slope * (x - x0) + y0;
  if (r !== r) {
    r = slope * (x - x1) + y1;
    if (r !== r && y0 === y1) r = y0;
  }
  return r;
}

/**
 * np.linspace(start, stop, num) (endpoint=True, float64): y = arange(num) * step + start with
 * step = (stop - start) / (num - 1), then y[-1] = stop exactly. A zero step (denormal delta) takes numpy's
 * other branch: y = (arange(num) / div) * delta + start.
 */
export function linspace(start: number, stop: number, num: number): Float64Array {
  const out = new Float64Array(num);
  const div = num - 1;
  const delta = stop - start;
  if (div > 0) {
    const step = delta / div;
    if (step === 0) {
      for (let i = 0; i < num; i++) out[i] = (i / div) * delta + start;
    } else {
      for (let i = 0; i < num; i++) out[i] = i * step + start;
    }
  } else {
    for (let i = 0; i < num; i++) out[i] = i * delta + start;
  }
  if (num > 1) out[num - 1] = stop;
  return out;
}

// ───────────────────────────── second-order sections ─────────────────────────────

/** Flattened sos ([b0, b1, b2, a0, a1, a2] per section) and zi ([z0, z1] per section) of one filter. */
export interface Sos {
  readonly sections: number;
  readonly sos: Float64Array;
  readonly zi: Float64Array;
}

export function makeSos(sos: readonly (readonly number[])[], zi: readonly (readonly number[])[]): Sos {
  if (sos.length !== zi.length) throw new RangeError('sos and zi have different section counts');
  const s = new Float64Array(6 * sos.length);
  const z = new Float64Array(2 * sos.length);
  sos.forEach((row, k) => {
    if (row.length !== 6 || zi[k].length !== 2) throw new RangeError('sos rows have 6 values, zi rows 2');
    for (let c = 0; c < 6; c++) s[6 * k + c] = row[c];
    z[2 * k] = zi[k][0];
    z[2 * k + 1] = zi[k][1];
  });
  return { sections: sos.length, sos: s, zi: z };
}

/**
 * One sample through every section (scipy _sosfilt.pyx _sosfilt_float, in this exact order):
 *   y  = b0 * x + z0;  z0 = (b1 * x - a1 * y) + z1;  z1 = b2 * x - a2 * y;  x = y
 * `z` holds the state of every section and is updated in place.
 */
export function sosStep(f: Sos, z: Float64Array, x: number): number {
  const s = f.sos;
  let xc = x;
  for (let k = 0; k < f.sections; k++) {
    const o = 6 * k;
    const q = 2 * k;
    const y = s[o] * xc + z[q];
    z[q] = s[o + 1] * xc - s[o + 4] * y + z[q + 1];
    z[q + 1] = s[o + 2] * xc - s[o + 5] * y;
    xc = y;
  }
  return xc;
}

/** z = zi * x0 (scipy: zi * x_0, the steady state scaled by a sample) */
export function scaledZi(f: Sos, x0: number, out: Float64Array = new Float64Array(f.zi.length)): Float64Array {
  for (let k = 0; k < f.zi.length; k++) out[k] = f.zi[k] * x0;
  return out;
}

/**
 * scipy.signal.sosfiltfilt(sos, x[a:a+n], padlen=min(padMax, n - 1)) with padtype 'odd', written into
 * out[a:a+n] (x and out may be the same array). Requires n >= 2 (the caller copies runs shorter than 3).
 *   ext = odd_ext(x, p): [2x0 - x[p], ..., 2x0 - x[1], x, 2x[n-1] - x[n-2], ..., 2x[n-1] - x[n-1-p]]
 *   forward sosfilt over ext with zi * ext[0]; backward over the reversed result with zi * (its last value);
 *   reverse and drop the pads.
 */
export function sosfiltfiltRun(f: Sos, x: ArrayLike<number>, a: number, n: number, out: Float64Array, padMax: number): void {
  const p = Math.min(padMax, n - 1);
  const m = n + 2 * p;
  const ext = new Float64Array(m);
  const x0 = x[a];
  const xl = x[a + n - 1];
  for (let i = 0; i < p; i++) ext[i] = 2 * x0 - x[a + p - i];
  for (let i = 0; i < n; i++) ext[p + i] = x[a + i];
  for (let i = 0; i < p; i++) ext[p + n + i] = 2 * xl - x[a + n - 2 - i];
  const z = scaledZi(f, ext[0]);
  for (let i = 0; i < m; i++) ext[i] = sosStep(f, z, ext[i]);
  scaledZi(f, ext[m - 1], z);
  for (let i = m - 1; i >= 0; i--) ext[i] = sosStep(f, z, ext[i]);
  for (let i = 0; i < n; i++) out[a + i] = ext[p + i];
}

// ───────────────────────────── fused multiply-add (OpenBLAS ddot order) ─────────────────────────────

const SPLITTER = 134217729; // 2^27 + 1 (Veltkamp)

/**
 * a * b + c rounded once, emulated with Dekker's exact product and an exact sum (no FMA instruction in JS).
 * The result is the correctly rounded value except in double-rounding cases of probability ~2^-50; finite,
 * non-overflowing inputs only (the pipeline's 3-vectors of about 10 m/s^2).
 */
export function fma(a: number, b: number, c: number): number {
  const p = a * b;
  let t = SPLITTER * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = SPLITTER * b;
  const bh = t - (t - b);
  const bl = b - bh;
  const e = ah * bh - p + ah * bl + al * bh + al * bl; // p + e = a * b exactly
  const s = p + c;
  const bb = s - p;
  const err = p - (s - bb) + (c - bb); // s + err = p + c exactly
  return s + (err + e);
}

/**
 * np.linalg.norm of a 1-D 3-vector: sqrt(x.dot(x)), which numpy hands to BLAS ddot. OpenBLAS's kernel on
 * this machine accumulates with fused multiply-adds: sqrt(fma(x2, x2, fma(x1, x1, x0 * x0))) (0 mismatches
 * in 1000 random vectors, against 10 % for the plain left-to-right sum). Used where Python calls
 * np.linalg.norm on a single vector (gyro_gravity's |g_pre|).
 */
export function norm3Dot(x0: number, x1: number, x2: number): number {
  return Math.sqrt(fma(x2, x2, fma(x1, x1, x0 * x0)));
}

/** np.linalg.norm(v, axis=-1) of a 3-vector inside an (n, 3) array: numpy reduces left to right. */
export function norm3(x0: number, x1: number, x2: number): number {
  return Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
}

/** angle_deg (preprocess.py:424-430) of two 3-vectors: degrees(atan2(|u x v|, u . v)). */
export function angleDeg3(u0: number, u1: number, u2: number, v0: number, v1: number, v2: number): number {
  // np.cross: cp0 = u1*v2 - u2*v1, cp1 = u2*v0 - u0*v2, cp2 = u0*v1 - u1*v0
  const c0 = u1 * v2 - u2 * v1;
  const c1 = u2 * v0 - u0 * v2;
  const c2 = u0 * v1 - u1 * v0;
  const c = Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2);
  const d = u0 * v0 + u1 * v1 + u2 * v2;
  return Math.atan2(c, d) * (180.0 / Math.PI);
}
