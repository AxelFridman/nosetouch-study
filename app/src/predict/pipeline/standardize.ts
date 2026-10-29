/**
 * stage2/standardize.py on the production path: stream_from_record (:155-207), canonical (:239-284),
 * time_normalised (:290-301), plus m4_detect.crop_stream / as_stream / window_for (m4_detect.py:233-256).
 * PORT_SPEC sections 1 and 2.3-2.7. Float64 everywhere; the window is rounded to float32 at the one point the
 * Python casts it (time_normalised's astype(np.float32)), with Math.fround.
 */
import type { ApproachWindow, Canonical, RawMotion, Rows3, Stream, StreamNormalization } from '../types';
import { FS_HZ, MIN_ROWS, N_CHANNELS, N_NORM, REST_MS, STILL_GYRO_DPS, YAW_CUM } from './constants';
import { gyroGravity } from './gravity';
import { angleDeg3, linspace, norm3, npInterp, searchLeft, searchRight } from './numeric';
import { quantizeLikeChrome, rotationRateToDevice, signRows } from './normalize';
import {
  fillGapsLinear,
  gridCount,
  gridTime,
  lowpassColumn,
  prepareTimes,
  repeatedSampleMask,
  resampleAt,
} from './preprocess';

/** Where the Python raises ValueError (types.ts: RangeError). */
export class PipelineError extends RangeError {
  constructor(
    readonly code: 'not_enough_data' | 'invalid_bounds' | 'too_few_samples',
    message: string,
  ) {
    super(message);
    this.name = 'PipelineError';
  }
}

/** The signed device-frame signals of a raw buffer: f (= accg), acc, gyro (normalize_motion), plus the raw rows the repeat rule reads. */
export interface SignedMotion {
  readonly accg: Rows3;
  readonly acc: Rows3;
  readonly gyro: Rows3;
  /** the raw rows after the optional iOS quantisation (repeated_sample_mask reads these) */
  readonly raw: { readonly acc: Rows3; readonly accg: Rows3; readonly rr: Rows3 };
}

/** stream_from_record steps 1-2: optional quantisation (WebKit only), then the session sign and the rate mapping. */
export function signedMotion(raw: RawMotion, norm: StreamNormalization): SignedMotion {
  let acc = raw.acc;
  let accg = raw.accg;
  let rr = raw.rr;
  if (norm.quantizeIos && norm.engine === 'webkit') {
    acc = quantizeLikeChrome(acc, 'acc');
    accg = quantizeLikeChrome(accg, 'acc');
    rr = quantizeLikeChrome(rr, 'rr');
  }
  return {
    acc: signRows(acc, norm.gravitySign),
    accg: signRows(accg, norm.gravitySign),
    gyro: rotationRateToDevice(rr, norm.rrVerdict, norm.rrUnits),
    raw: { acc, accg, rr },
  };
}

/**
 * standardize.stream_from_record(record, norm) steps 1-4 (production defaults: no mirror, no screen flip, no
 * calibration magnitude, no baseline gravity):
 *   1. sign and rate mapping (signedMotion);
 *   2. drop isolated repeated samples (repeated_sample_mask on the raw rows; fewer than 5 kept -> error);
 *   3. resample every one of the 9 columns [accg, acc, gyro] from its own finite samples onto the 60 Hz grid
 *      t0 + i * step through the last kept time (NaN inside gaps > 100 ms); valid = f and gyro finite;
 *   4. zero-phase Butterworth per finite run.
 * Throws PipelineError('not_enough_data') where Python raises ValueError.
 */
export function streamFrom(raw: RawMotion, norm: StreamNormalization): Stream {
  const nRaw = raw.t.length;
  const sig = signedMotion(raw, norm);
  let drop: Uint8Array = new Uint8Array(nRaw);
  if (nRaw > 1) drop = repeatedSampleMask(sig.raw, nRaw, norm.engine);
  const keptRows: number[] = [];
  for (let i = 0; i < nRaw; i++) if (!drop[i]) keptRows.push(i);
  if (keptRows.length < MIN_ROWS) throw new PipelineError('not_enough_data', `fewer than ${MIN_ROWS} motion samples`);
  const rows = prepareTimes(raw.t, keptRows);
  const ng = rows.length ? gridCount(raw.t[rows[0]], raw.t[rows[rows.length - 1]]) : 0;
  const tg = new Float64Array(ng);
  for (let i = 0; i < ng; i++) tg[i] = gridTime(raw.t[rows[0]], i);
  const blocks = [sig.accg, sig.acc, sig.gyro];
  const out = [new Float64Array(3 * ng), new Float64Array(3 * ng), new Float64Array(3 * ng)];
  const valid = new Uint8Array(ng).fill(1);
  const tv = new Float64Array(rows.length);
  const xv = new Float64Array(rows.length);
  const col = new Float64Array(ng);
  for (let b = 0; b < 3; b++) {
    for (let c = 0; c < 3; c++) {
      let len = 0;
      for (const i of rows) {
        const v = blocks[b].data[3 * i + c];
        if (Number.isFinite(v)) {
          tv[len] = raw.t[i];
          xv[len] = v;
          len++;
        }
      }
      let r = 0;
      for (let i = 0; i < ng; i++) {
        const tn = tg[i];
        while (r < len && tv[r] < tn) r++;
        col[i] = len ? resampleAt(tv, xv, len, tn, r) : NaN;
      }
      if (b !== 1) for (let i = 0; i < ng; i++) if (!Number.isFinite(col[i])) valid[i] = 0; // f and gyro, before the low-pass
      lowpassColumn(col, ng);
      for (let i = 0; i < ng; i++) out[b][3 * i + c] = col[i];
    }
  }
  return {
    t: tg,
    f: { n: ng, data: out[0] },
    acc: { n: ng, data: out[1] },
    gyro: { n: ng, data: out[2] },
    valid,
    fsHz: FS_HZ,
    engine: norm.engine,
    repeatsDropped: nRaw - keptRows.length,
    tRaw: raw.t.slice(),
    gravityMagnitude: null,
    baselineGravity: null,
  };
}

/** Rows of a stream whose t lies in [startMs, stopMs] (m4_detect.crop_stream), as m4_detect.as_stream builds them. */
export function cropStream(stream: Stream, startMs: number, stopMs: number): Stream {
  const t = stream.t;
  // a stream grid is finite and strictly increasing, so {i : start <= t[i] <= stop} is one index range
  // (NaN or reversed bounds give the empty range, like the Python mask)
  const lo = searchLeft(t, startMs);
  const hi = Math.max(lo, searchRight(t, stopMs));
  const pick = (r: Rows3): Rows3 => ({ n: hi - lo, data: r.data.slice(3 * lo, 3 * hi) });
  return {
    t: t.slice(lo, hi),
    f: pick(stream.f),
    acc: pick(stream.acc),
    gyro: pick(stream.gyro),
    valid: stream.valid.slice(lo, hi),
    // as_stream: fs_hz = FS_HZ, no calibration magnitude, no baseline gravity, no engine / raw times
    fsHz: FS_HZ,
    engine: null,
    repeatsDropped: 0,
    tRaw: new Float64Array(0),
    gravityMagnitude: null,
    baselineGravity: null,
  };
}

/** canonical()'s result plus the gravity details the reference reports. */
export interface CanonicalDetail extends Canonical {
  /** the magnitude every g row was rescaled to (|g_pre| when plausible, else 9.80665) */
  readonly gravityMagnitude: number;
  readonly gyroMissingSteps: number;
}

export interface CanonicalOptions {
  readonly stillDps?: number;
  readonly restMs?: number;
}

/**
 * standardize.canonical(stream, onset_ms, end_ms): gravity_v1 from the rest windows around the bounds, then the 17
 * channels on the stream's grid (X[c * n + i]). yaw_cum and tilt_from_start are referenced to the onset sample.
 * Throws PipelineError where Python raises ValueError (non-finite or reversed bounds, fewer than 3 samples).
 */
export function canonical(stream: Stream, onsetMs: number, endMs: number, opts: CanonicalOptions = {}): CanonicalDetail {
  const t = stream.t;
  const n = t.length;
  if (!(Number.isFinite(onsetMs) && Number.isFinite(endMs)) || endMs <= onsetMs) {
    throw new PipelineError('invalid_bounds', `invalid bounds ${onsetMs}..${endMs}`);
  }
  let iOn = searchLeft(t, onsetMs);
  let iEnd = searchRight(t, endMs) - 1;
  iOn = Math.max(iOn, 0);
  iEnd = Math.min(iEnd, n - 1);
  if (iEnd - iOn < 2) {
    throw new PipelineError('too_few_samples', `bounds ${onsetMs.toFixed(0)}..${endMs.toFixed(0)} ms hold fewer than 3 grid samples`);
  }
  const f = fillGapsLinear(t, stream.f.data, n);
  const w = fillGapsLinear(t, stream.gyro.data, n);
  const gg = gyroGravity(t, f, w, n, iOn, iEnd, {
    stillDps: opts.stillDps ?? STILL_GYRO_DPS,
    restMs: opts.restMs ?? REST_MS,
    magnitude: stream.gravityMagnitude,
    gPreFallback: stream.baselineGravity,
  });
  const g = gg.g;
  const X = new Float64Array(N_CHANNELS * n);
  const ch = (c: number, i: number, v: number): void => {
    X[c * n + i] = v;
  };
  const u = new Float64Array(3 * n);
  const wv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const o = 3 * i;
    // u = unit_rows(g) (a zero or NaN norm gives NaN)
    const gn = norm3(g[o], g[o + 1], g[o + 2]);
    const u0 = gn > 0 ? g[o] / gn : NaN;
    const u1 = gn > 0 ? g[o + 1] / gn : NaN;
    const u2 = gn > 0 ? g[o + 2] / gn : NaN;
    u[o] = u0;
    u[o + 1] = u1;
    u[o + 2] = u2;
    const a0 = f[o] - g[o];
    const a1 = f[o + 1] - g[o + 1];
    const a2 = f[o + 2] - g[o + 2];
    const w0 = w[o];
    const w1 = w[o + 1];
    const w2 = w[o + 2];
    const yr = w0 * u0 + w1 * u1 + w2 * u2; // np.sum(w * u, axis=1)
    const av = a0 * u0 + a1 * u1 + a2 * u2;
    wv[i] = yr;
    ch(0, i, a0);
    ch(1, i, a1);
    ch(2, i, a2);
    ch(3, i, w0);
    ch(4, i, w1);
    ch(5, i, w2);
    ch(6, i, u0);
    ch(7, i, u1);
    ch(8, i, u2);
    ch(9, i, norm3(w0, w1, w2));
    ch(10, i, norm3(a0, a1, a2));
    ch(11, i, av);
    ch(12, i, norm3(a0 - av * u0, a1 - av * u1, a2 - av * u2));
    ch(13, i, yr);
    ch(14, i, norm3(w0 - yr * u0, w1 - yr * u1, w2 - yr * u2));
  }
  // yaw = cumulative_trapezoid(wv, t / 1000, initial=0): cumsum of (dts * (wv[k+1] + wv[k])) / 2.0, then minus yaw[i_on]
  const yaw = new Float64Array(n);
  for (let k = 0; k + 1 < n; k++) {
    const term = ((t[k + 1] / 1000.0 - t[k] / 1000.0) * (wv[k + 1] + wv[k])) / 2.0;
    yaw[k + 1] = k === 0 ? term : yaw[k] + term;
  }
  const yawOn = yaw[iOn];
  const v0 = u[3 * iOn];
  const v1 = u[3 * iOn + 1];
  const v2 = u[3 * iOn + 2];
  for (let i = 0; i < n; i++) {
    ch(15, i, yaw[i] - yawOn);
    ch(16, i, angleDeg3(u[3 * i], u[3 * i + 1], u[3 * i + 2], v0, v1, v2));
  }
  return {
    t,
    X,
    n,
    g: { n, data: g },
    valid: stream.valid,
    onsetMs,
    endMs,
    iOn,
    iEnd,
    closureDeg: gg.closureDeg,
    restPreN: gg.preN,
    restPostN: gg.postN,
    gravityMethod: gg.method,
    gravityMagnitude: gg.magnitude,
    gyroMissingSteps: gg.gyroMissingSteps,
  };
}

/**
 * standardize.time_normalised(can, start_ms, end_ms, n=64): every channel at np.linspace(start, end, n) by
 * np.interp on the canonical grid (edge values outside), yaw_cum minus its first value, then float32.
 */
export function timeNormalised(can: Canonical, startMs: number, endMs: number, nOut = N_NORM): ApproachWindow {
  const grid = linspace(startMs, endMs, nOut);
  const n = can.n;
  const Xn = new Float64Array(N_CHANNELS * nOut);
  for (let c = 0; c < N_CHANNELS; c++) {
    const fp = can.X.subarray(c * n, (c + 1) * n);
    for (let k = 0; k < nOut; k++) Xn[c * nOut + k] = npInterp(grid[k], can.t, fp, n);
  }
  const y0 = Xn[YAW_CUM * nOut];
  for (let k = 0; k < nOut; k++) Xn[YAW_CUM * nOut + k] -= y0;
  const data = new Float32Array(N_CHANNELS * nOut);
  for (let k = 0; k < data.length; k++) data[k] = Math.fround(Xn[k]);
  return { data, onsetMs: startMs, endMs };
}

/**
 * m4_detect.window_for(st, onset_ms, end_ms, crop_s, post_s): crop the stream to [onset - crop_s, end + post_s]
 * (seconds; post_s defaults to crop_s), canonical() on the crop, time_normalised over the bounds. Lane m4 calls it
 * with crop_s = 1.0 and post_s = lookahead_s = 0.5, so the window uses no sample after the bout's decision time.
 */
export function windowFor(
  stream: Stream,
  onsetMs: number,
  endMs: number,
  cropS = 1.0,
  postS?: number,
): { window: ApproachWindow; can: CanonicalDetail; crop: Stream } {
  const post = postS ?? cropS;
  const crop = cropStream(stream, onsetMs - cropS * 1000.0, endMs + post * 1000.0);
  const can = canonical(crop, onsetMs, endMs);
  return { window: timeNormalised(can, onsetMs, endMs), can, crop };
}

