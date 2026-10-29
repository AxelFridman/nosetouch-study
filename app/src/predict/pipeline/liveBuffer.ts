/**
 * The growing raw buffer of one live stream (types.ts LiveBuffer, APP_SPEC section 1.3).
 *
 * Every sample is stored exactly as core/recorder.ts StreamRecorder.addMotionRel stores it, with the collector's
 * own roundValue and ROUNDING (imported, never copied): t = round(t - t0, 2) where t0 is the first accepted
 * sample's t, acc / accg components to 1e-4, rotationRate alpha / beta / gamma to 1e-3, and a missing or
 * non-finite component becomes NaN (the recorder stores null per component; core/sensors.ts already nulls a
 * whole vector when one component is missing). raw() therefore equals rawMotionFromRecord() of the record the
 * collector would have stored for the same samples (liveBuffer.test.ts checks it against StreamRecorder).
 *
 * The one difference from the recorder: a sample whose stored t is not finite or not after the previous stored t
 * is rejected (push returns -1). The recorder keeps such a row; the browser clock never produces one.
 *
 * Storage is append-only and grows by doubling, so a RawMotion view handed out earlier stays valid (its rows are
 * never rewritten); clear() starts new arrays for the same reason.
 */
import { ROUNDING } from '../../core/constants';
import { roundValue } from '../../core/recorder';
import type { LiveBuffer, LiveSample, RawMotion } from '../types';

const INITIAL_CAPACITY = 1024;

function grow(a: Float64Array<ArrayBuffer>, need: number): Float64Array<ArrayBuffer> {
  if (need <= a.length) return a;
  let cap = Math.max(a.length, 1);
  while (cap < need) cap *= 2;
  const b = new Float64Array(cap);
  b.set(a);
  return b;
}

function stored(v: number | null | undefined, decimals: number): number {
  const r = roundValue(v, decimals);
  return r === null ? NaN : r;
}

export function createLiveBuffer(): LiveBuffer {
  let t = new Float64Array(INITIAL_CAPACITY);
  let acc = new Float64Array(3 * INITIAL_CAPACITY);
  let accg = new Float64Array(3 * INITIAL_CAPACITY);
  let rr = new Float64Array(3 * INITIAL_CAPACITY);
  let n = 0;
  let t0: number | null = null;

  const buffer: LiveBuffer = {
    push(s: LiveSample): number {
      if (typeof s.t !== 'number' || !Number.isFinite(s.t)) return -1;
      const origin = t0 ?? s.t;
      const tr = roundValue(s.t - origin, ROUNDING.time); // StreamRecorder.rel
      if (tr === null) return -1;
      if (n > 0 && !(tr > t[n - 1])) return -1;
      if (t0 === null) t0 = s.t;
      t = grow(t, n + 1);
      acc = grow(acc, 3 * (n + 1));
      accg = grow(accg, 3 * (n + 1));
      rr = grow(rr, 3 * (n + 1));
      t[n] = tr;
      const k = 3 * n;
      // push3(cols, v, ROUNDING.acc): each component rounded on its own
      acc[k] = stored(s.acc?.[0], ROUNDING.acc);
      acc[k + 1] = stored(s.acc?.[1], ROUNDING.acc);
      acc[k + 2] = stored(s.acc?.[2], ROUNDING.acc);
      accg[k] = stored(s.accg?.[0], ROUNDING.acc);
      accg[k + 1] = stored(s.accg?.[1], ROUNDING.acc);
      accg[k + 2] = stored(s.accg?.[2], ROUNDING.acc);
      rr[k] = stored(s.rr?.alpha, ROUNDING.rr);
      rr[k + 1] = stored(s.rr?.beta, ROUNDING.rr);
      rr[k + 2] = stored(s.rr?.gamma, ROUNDING.rr);
      return n++;
    },
    get n(): number {
      return n;
    },
    get t0(): number | null {
      return t0;
    },
    raw(count?: number): RawMotion {
      const k = count === undefined ? n : Math.max(0, Math.min(Math.floor(count), n));
      return {
        t: t.subarray(0, k),
        acc: { n: k, data: acc.subarray(0, 3 * k) },
        accg: { n: k, data: accg.subarray(0, 3 * k) },
        rr: { n: k, data: rr.subarray(0, 3 * k) },
      };
    },
    clear(): void {
      t = new Float64Array(INITIAL_CAPACITY);
      acc = new Float64Array(3 * INITIAL_CAPACITY);
      accg = new Float64Array(3 * INITIAL_CAPACITY);
      rr = new Float64Array(3 * INITIAL_CAPACITY);
      n = 0;
      t0 = null;
    },
  };
  return buffer;
}
