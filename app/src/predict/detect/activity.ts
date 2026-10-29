/**
 * The causal activity signal of lane m4's detector: |w| (deg/s) of the standardised stream, gaps filled, low-passed
 * by a one-pass 2nd-order Butterworth (m4_detect.activity, analysis/stage2/lanes/m4_detect.py:105-117).
 *
 *  - activity(): the batch function, an exact port (the reference the parity tests compare);
 *  - ActivityFilter: the same signal fed sample by sample (O(1) per sample). On a complete stream of at least
 *    MIN_ACTIVITY_N samples it releases exactly activity()'s values, in order.
 *
 * The filter coefficients come from the exported bundle (DetectorBundle.activitySos / activityZi, computed by
 * scipy: butter(2, smooth_hz, fs=60, output='sos') and sosfilt_zi); they are never designed at runtime.
 */
import type { Rows3, SegParams } from '../types';
import { fillGapsRows, norm3, scaledState, sosfiltInPlace, sosfiltStep } from './numeric';

/** m4_detect.activity returns the UNFILTERED |w| for streams shorter than this (m4_detect.py:110). */
export const MIN_ACTIVITY_N = 12;

/** The causal activity filter as exported (scipy sos rows [b0, b1, b2, a0, a1, a2] and sosfilt_zi rows). */
export interface ActivityFilterCoefs {
  readonly sos: readonly (readonly number[])[];
  readonly zi: readonly (readonly number[])[];
}

function checkCausal(p: SegParams): void {
  // m4_detect.py:112-117: production uses the causal variant; the zero-phase one (cuefree v0) is not exported.
  if (!p.causal) throw new Error('detect: only the causal activity filter is supported (SegParams.causal = false)');
}

/**
 * m4_detect.activity(t, gyro, p) with p.causal: fill_gaps_linear per column, row norm (non-finite -> 0), then
 * sosfilt with the initial state zi * s[0]. Streams shorter than 12 samples (or smooth_hz = 0) are returned
 * unfiltered, as in Python.
 */
export function activity(t: ArrayLike<number>, gyro: Rows3, p: SegParams, coefs: ActivityFilterCoefs): Float64Array {
  checkCausal(p);
  const n = gyro.n;
  const w = fillGapsRows(t, gyro).data;
  const s = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = norm3(w[3 * i], w[3 * i + 1], w[3 * i + 2]);
    s[i] = Number.isFinite(v) ? v : 0.0; // m4_detect.py:109
  }
  if (!p.smooth_hz || n < MIN_ACTIVITY_N) return s; // m4_detect.py:110-111
  sosfiltInPlace(coefs.sos, s, scaledState(coefs.zi, s[0])); // m4_detect.py:113-116 (one pass, causal)
  return s;
}

/** One released activity sample: grid time (ms) and the filtered |w| (deg/s). */
export interface ActivitySample {
  readonly t: number;
  readonly act: number;
}

/**
 * activity() fed sample by sample.
 *
 * A row whose gyro is missing (NaN inside a resampling gap > 100 ms) cannot be filled before the next finite value
 * of its column arrives (fill_gaps_linear interpolates between both neighbours), so rows are released in order as
 * soon as every column has a finite value at or after them; finish() releases the rest exactly as
 * fill_gaps_linear does at the end of a stream (the last finite value is held; a column that never had one gives
 * activity 0). The filter starts at the first released row with the state zi * s[0].
 *
 * Equality with activity(): identical values on a complete stream of >= MIN_ACTIVITY_N samples (activity() leaves
 * shorter streams unfiltered; nothing can be detected in them anyway).
 */
export class ActivityFilter {
  private z: Float64Array | null = null;
  private readonly lastT = [NaN, NaN, NaN];
  private readonly lastX = [NaN, NaN, NaN];
  /** pending rows (not yet released): time and the three columns (NaN = not filled yet) */
  private readonly pt: number[] = [];
  private readonly px: number[][] = [];
  private released = 0;

  constructor(
    private readonly p: SegParams,
    private readonly coefs: ActivityFilterCoefs,
  ) {
    checkCausal(p);
  }

  /** number of samples released so far */
  get count(): number {
    return this.released;
  }

  /** Adds one grid row (t ms, gyro deg/s about device x, y, z; NaN = missing) and returns the released samples. */
  push(t: number, gx: number, gy: number, gz: number): ActivitySample[] {
    const row = [gx, gy, gz];
    this.pt.push(t);
    this.px.push(row.slice());
    const k = this.pt.length - 1;
    for (let j = 0; j < 3; j++) {
      const x = row[j];
      if (!Number.isFinite(x)) continue;
      // fill this column's pending NaN rows (they are the rows since the column's last finite value)
      const t0 = this.lastT[j];
      const x0 = this.lastX[j];
      for (let r = 0; r < k; r++) {
        if (Number.isFinite(this.px[r][j])) continue;
        if (Number.isNaN(t0)) {
          this.px[r][j] = x; // leading NaN run: np.interp left edge holds the first finite value
        } else {
          // np.interp between the neighbouring finite samples (t0, x0) and (t, x): slope * (tr - t0) + x0
          const slope = (x - x0) / (t - t0);
          this.px[r][j] = slope * (this.pt[r] - t0) + x0;
        }
      }
      this.lastT[j] = t;
      this.lastX[j] = x;
    }
    return this.release(false);
  }

  /** End of the stream: releases every pending row (trailing NaN rows hold the last finite value). */
  finish(): ActivitySample[] {
    for (let r = 0; r < this.pt.length; r++) {
      for (let j = 0; j < 3; j++) {
        if (!Number.isFinite(this.px[r][j]) && !Number.isNaN(this.lastT[j])) this.px[r][j] = this.lastX[j];
      }
    }
    return this.release(true);
  }

  private release(all: boolean): ActivitySample[] {
    const out: ActivitySample[] = [];
    let r = 0;
    while (r < this.pt.length) {
      const x = this.px[r];
      if (!all && !(Number.isFinite(x[0]) && Number.isFinite(x[1]) && Number.isFinite(x[2]))) break;
      const v = norm3(x[0], x[1], x[2]);
      const s = Number.isFinite(v) ? v : 0.0;
      let a = s;
      if (this.p.smooth_hz) {
        if (this.z === null) this.z = scaledState(this.coefs.zi, s);
        a = sosfiltStep(this.coefs.sos, s, this.z);
      }
      out.push({ t: this.pt[r], act: a });
      r++;
    }
    if (r > 0) {
      this.pt.splice(0, r);
      this.px.splice(0, r);
      this.released += r;
    }
    return out;
  }
}
