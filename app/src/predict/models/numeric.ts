/**
 * Numeric primitives of the Python reference, with numpy's exact semantics where the result depends on them
 * (docs/stage3/PORT_SPEC.md section 5.1). Float64 throughout; every function is pure.
 */

/** numpy.finfo(float64).max: what numpy.nan_to_num maps +inf to (and -inf to its negative). */
export const DBL_MAX = Number.MAX_VALUE;

/** Canonical channels in standardize.CHANNELS order: the row index of every (17, 64) window. */
export const CHANNELS = [
  'acc_x',
  'acc_y',
  'acc_z',
  'gyro_x',
  'gyro_y',
  'gyro_z',
  'up_x',
  'up_y',
  'up_z',
  'gyro_mag',
  'acc_mag',
  'acc_vert',
  'acc_horiz',
  'yaw_rate',
  'gyro_horiz',
  'yaw_cum',
  'tilt_from_start',
] as const;

/** standardize.time_normalised's N_NORM (config.py:116): samples per approach window. */
export const N_NORM = 64;

/** Row index of a canonical channel (throws on an unknown name: a model file that does not match the app). */
export function channelIndex(name: string): number {
  const i = (CHANNELS as readonly string[]).indexOf(name);
  if (i < 0) throw new RangeError(`unknown canonical channel ${JSON.stringify(name)}`);
  return i;
}

/** numpy.clip for a scalar: NaN propagates (np.clip is minimum(maximum(x, lo), hi)). */
export function npClip(x: number, lo: number, hi: number): number {
  if (x !== x) return x;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/** numpy.nan_to_num for a scalar: NaN -> 0, +inf -> DBL_MAX, -inf -> -DBL_MAX. */
export function nanToNum(x: number): number {
  if (x !== x) return 0;
  if (x === Infinity) return DBL_MAX;
  if (x === -Infinity) return -DBL_MAX;
  return x;
}

/**
 * 1 / (1 + exp(-z)) with no clip: final_infer.sigmoid(z) (clip None), m1_models.sigmoid and
 * scipy.special.expit (scikit-learn's binary LogisticRegression.predict_proba).
 */
export function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-z));
}

/** final_infer.sigmoid(z, clip) and m4_models.sigmoid (clip 40): 1 / (1 + exp(-clip(z, -c, c))). */
export function sigmoidClipped(z: number, c: number): number {
  return 1.0 / (1.0 + Math.exp(-npClip(z, -c, c)));
}

/** final_infer.logit (final_infer.py:36-38): ln(p) - log1p(-p) with p clipped to [eps, 1 - eps] (eps 1e-12). */
export function logitLog1p(p: number, eps: number): number {
  const q = npClip(p, eps, 1 - eps);
  return Math.log(q) - Math.log1p(-q);
}

/**
 * m4_models.logit (m4_models.py:46-48) and m1_models.logit (m1_models.py:76-78): ln(p / (1 - p)) with p clipped to
 * [eps, 1 - eps] (EPS 1e-6 in both). Not the same expression as final_infer.logit: keep them apart.
 */
export function logitRatio(p: number, eps: number): number {
  const q = npClip(p, eps, 1 - eps);
  return Math.log(q / (1 - q));
}

/**
 * numpy.interp(x, xp, fp) for one x (numpy _core/src/multiarray/compiled_base.c arr_interp): xp increasing;
 * NaN x gives NaN; below xp[0] gives fp[0], above xp[last] gives fp[last]; x == xp[last] gives fp[last];
 * otherwise j with xp[j] <= x < xp[j + 1]: fp[j] when x == xp[j], else slope * (x - xp[j]) + fp[j] with
 * slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]); a NaN result is retried from the right end, then
 * fp[j] when fp[j] == fp[j + 1]. A single xp gives fp[0].
 */
export function npInterp(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>): number {
  const n = xp.length;
  if (n === 0 || fp.length !== n) throw new RangeError('npInterp: xp and fp must have the same, non-zero length');
  if (x !== x) return x;
  if (n === 1) return fp[0];
  if (x > xp[n - 1]) return fp[n - 1];
  if (x < xp[0]) return fp[0];
  // j = (first index with xp[i] > x) - 1, numpy's binary_search_with_guess for an increasing xp
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = lo + ((hi - lo) >> 1);
    if (x >= xp[mid]) lo = mid + 1;
    else hi = mid;
  }
  const j = lo - 1;
  if (j === n - 1) return fp[j];
  if (xp[j] === x) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  let r = slope * (x - xp[j]) + fp[j];
  if (r !== r) {
    r = slope * (x - xp[j + 1]) + fp[j + 1];
    if (r !== r && fp[j] === fp[j + 1]) r = fp[j];
  }
  return r;
}

/**
 * numpy's pairwise summation (_core/src/umath/loops_utils.h.src pairwise_sum) of a[off], a[off + stride], ...
 * (n terms): a plain loop below 8 terms, eight interleaved accumulators up to 128, halves above. np.sum and
 * np.mean along a contiguous axis use it (the final model's time bins: blocks of 8, checked against numpy).
 */
export function pairwiseSum(a: ArrayLike<number>, off: number, n: number, stride = 1): number {
  if (n < 8) {
    let res = -0.0;
    for (let i = 0; i < n; i++) res += a[off + i * stride];
    return res;
  }
  if (n <= 128) {
    let r0 = a[off];
    let r1 = a[off + stride];
    let r2 = a[off + 2 * stride];
    let r3 = a[off + 3 * stride];
    let r4 = a[off + 4 * stride];
    let r5 = a[off + 5 * stride];
    let r6 = a[off + 6 * stride];
    let r7 = a[off + 7 * stride];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      const b = off + i * stride;
      r0 += a[b];
      r1 += a[b + stride];
      r2 += a[b + 2 * stride];
      r3 += a[b + 3 * stride];
      r4 += a[b + 4 * stride];
      r5 += a[b + 5 * stride];
      r6 += a[b + 6 * stride];
      r7 += a[b + 7 * stride];
    }
    // ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), numpy's order
    const left = r0 + r1 + (r2 + r3);
    const right = r4 + r5 + (r6 + r7);
    let res = left + right;
    for (; i < n; i++) res += a[off + i * stride];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, off, n2, stride) + pairwiseSum(a, off + n2 * stride, n - n2, stride);
}

/** Python's round() of a float to an integer: half to even (int(round(x))). */
export function roundHalfEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const f = Math.floor(x);
  const d = x - f; // exact for |x| < 2^52
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** np.linspace(start, stop, num) (endpoint True): start + i * step, the last point set to stop exactly. */
export function npLinspace(start: number, stop: number, num: number): number[] {
  if (num === 1) return [start];
  const div = num - 1;
  const step = (stop - start) / div;
  const out: number[] = [];
  for (let i = 0; i < num; i++) out.push(start + i * step);
  out[num - 1] = stop;
  return out;
}

/** Every value is finite (no NaN, no infinity). */
export function isFiniteAll(a: ArrayLike<number>): boolean {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false;
  return true;
}
