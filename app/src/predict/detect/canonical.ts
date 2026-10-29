/**
 * The window path of one bout: m4_detect.window_for = crop_stream -> as_stream -> standardize.canonical ->
 * standardize.time_normalised (analysis/stage2/lanes/m4_detect.py:242-256; analysis/stage2/standardize.py:239-301;
 * gravity_v1 in analysis/motionlab/preprocess.py:376-544).
 *
 * OWNERSHIP NOTE. These three functions belong to pipeline/ (PipelineApi.cropStream / canonical / timeNormalised).
 * detect/ only needs them through the WindowPipeline seam below; this file is detect's own port of the same Python,
 * so that the detector, the segment features and the attempt rule can be checked against the golden fixtures
 * end to end before pipeline/ is wired. The integrator keeps ONE implementation (both must pass the same fixture
 * checks: windows within 1 float32 ulp, features rel 1e-8).
 *
 * Production path only: as_stream drops the calibration magnitude and the baseline gravity (m4_detect.py:233-239),
 * so gyro_gravity runs with magnitude = None and no fallback, still_dps = STILL_GYRO_DPS (20), rest_ms = REST_MS (300).
 */
import type { ApproachWindow, Canonical, CanonicalChannel, PipelineApi, Rows3, Stream } from '../types';
import { fillGapsRows, npInterp, searchsortedLeft, searchsortedRight } from './numeric';

/** The pipeline functions detect/ needs to cut a bout's window (PipelineApi subset). */
export type WindowPipeline = Pick<PipelineApi, 'cropStream' | 'canonical' | 'timeNormalised'>;

/** standardize.CHANNELS (row order of every (17, n) array). */
export const CHANNELS: readonly CanonicalChannel[] = [
  'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'up_x', 'up_y', 'up_z',
  'gyro_mag', 'acc_mag', 'acc_vert', 'acc_horiz', 'yaw_rate', 'gyro_horiz', 'yaw_cum', 'tilt_from_start',
];
export const CH: Readonly<Record<CanonicalChannel, number>> = Object.fromEntries(
  CHANNELS.map((c, i) => [c, i]),
) as Record<CanonicalChannel, number>;

export const N_NORM = 64;
const STILL_GYRO_DPS = 20.0; // stage2/config.py:114
const REST_MS = 300.0; // stage2/config.py:115
const MIN_REST_N = 5; // preprocess.py:475
const GRAVITY = 9.80665; // motionlab/conventions.py:66
const DEG2RAD = Math.PI / 180.0; // motionlab/conventions.py:67
const RAD2DEG_NP = 180.0 / Math.PI; // np.degrees: x * (180 / pi)

// ───────────────────────────── crop ─────────────────────────────

/** m4_detect.crop_stream + as_stream: the samples with startMs <= t <= stopMs (both inclusive). */
export function cropStream(stream: Stream, startMs: number, stopMs: number): Stream {
  const t = stream.t;
  let i0 = 0;
  while (i0 < t.length && !(t[i0] >= startMs)) i0++;
  let i1 = i0;
  while (i1 < t.length && t[i1] <= stopMs) i1++;
  // t is ascending, so the selection (t >= start) & (t <= stop) is the contiguous range [i0, i1)
  const rows = (r: Rows3): Rows3 => ({ n: i1 - i0, data: r.data.slice(3 * i0, 3 * i1) });
  return {
    t: t.slice(i0, i1),
    f: rows(stream.f),
    acc: rows(stream.acc),
    gyro: rows(stream.gyro),
    valid: stream.valid.slice(i0, i1),
    fsHz: stream.fsHz,
    engine: stream.engine,
    repeatsDropped: 0,
    tRaw: t.slice(i0, i1),
    gravityMagnitude: null, // as_stream drops both (m4_detect.py:233-239)
    baselineGravity: null,
  };
}

// ───────────────────────────── gyro integration (preprocess.py:376-418) ─────────────────────────────

/**
 * integrate_gyro(t_ms, w_dps): R[k] (row-major 3x3 at 9k) maps device-frame vectors at sample k into the device
 * frame of sample 0 (midpoint rate per step; a step with a missing rate or dt rotates by zero).
 * `w` holds n rows starting at row `w0` (so a sub-range of a Rows3 can be integrated without a copy).
 */
export function integrateGyro(tMs: ArrayLike<number>, w: Float64Array, w0: number, n: number): Float64Array {
  const R = new Float64Array(9 * n);
  let x = 0.0;
  let y = 0.0;
  let z = 0.0;
  let qw = 1.0;
  const put = (k: number): void => {
    // quaternion_to_matrix (preprocess.py:304-319): normalise again, then the standard [x, y, z, w] matrix
    const nq = Math.sqrt(x * x + y * y + z * z + qw * qw);
    const a = x / nq;
    const b = y / nq;
    const c = z / nq;
    const d = qw / nq;
    const o = 9 * k;
    R[o] = 1 - 2 * (b * b + c * c);
    R[o + 1] = 2 * (a * b - c * d);
    R[o + 2] = 2 * (a * c + b * d);
    R[o + 3] = 2 * (a * b + c * d);
    R[o + 4] = 1 - 2 * (a * a + c * c);
    R[o + 5] = 2 * (b * c - a * d);
    R[o + 6] = 2 * (a * c - b * d);
    R[o + 7] = 2 * (b * c + a * d);
    R[o + 8] = 1 - 2 * (a * a + b * b);
  };
  if (n > 0) put(0);
  for (let k = 0; k < n - 1; k++) {
    const dt = tMs[k + 1] / 1000.0 - tMs[k] / 1000.0;
    const p = 3 * (w0 + k);
    const a0 = w[p] * DEG2RAD;
    const a1 = w[p + 1] * DEG2RAD;
    const a2 = w[p + 2] * DEG2RAD;
    const b0 = w[p + 3] * DEG2RAD;
    const b1 = w[p + 4] * DEG2RAD;
    const b2 = w[p + 5] * DEG2RAD;
    if (
      !(Number.isFinite(a0) && Number.isFinite(a1) && Number.isFinite(a2)) ||
      !(Number.isFinite(b0) && Number.isFinite(b1) && Number.isFinite(b2)) ||
      !Number.isFinite(dt)
    ) {
      put(k + 1);
      continue;
    }
    const ox = (a0 + b0) * 0.5 * dt;
    const oy = (a1 + b1) * 0.5 * dt;
    const oz = (a2 + b2) * 0.5 * dt;
    const th = Math.sqrt(ox * ox + oy * oy + oz * oz);
    if (th < 1e-15) {
      put(k + 1);
      continue;
    }
    const h = 0.5 * th;
    const s = Math.sin(h) / th;
    const dx = ox * s;
    const dy = oy * s;
    const dz = oz * s;
    const dw = Math.cos(h);
    // q <- q (x) dq (body-frame increment), preprocess.py:409-414
    const nx = qw * dx + x * dw + y * dz - z * dy;
    const ny = qw * dy - x * dz + y * dw + z * dx;
    const nz = qw * dz + x * dy - y * dx + z * dw;
    const nw = qw * dw - x * dx - y * dy - z * dz;
    const nq = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
    x = nx / nq;
    y = ny / nq;
    z = nz / nq;
    qw = nw / nq;
    put(k + 1);
  }
  return R;
}

// ───────────────────────────── gravity_v1 (preprocess.py:465-544) ─────────────────────────────

interface Gravity {
  g: Float64Array;
  preN: number;
  postN: number;
  closureDeg: number;
  method: 'fwd_bwd' | 'fwd_only';
}

/** angle_deg (preprocess.py:424-430) of two 3-vectors: degrees(atan2(|u x v|, u . v)). */
export function angleDeg(u0: number, u1: number, u2: number, v0: number, v1: number, v2: number): number {
  const c0 = u1 * v2 - u2 * v1;
  const c1 = u2 * v0 - u0 * v2;
  const c2 = u0 * v1 - u1 * v0;
  const c = Math.sqrt(c0 * c0 + c1 * c1 + c2 * c2);
  const d = u0 * v0 + u1 * v1 + u2 * v2;
  return Math.atan2(c, d) * RAD2DEG_NP;
}

function gyroGravity(t: Float64Array, F: Float64Array, W: Float64Array, iOn: number, iEnd: number): Gravity {
  const n = t.length;
  const R = integrateGyro(t, W, 0, n);
  const Fw = new Float64Array(3 * n); // R[k] @ f[k]: specific force in the device frame of sample 0
  const wmag = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const o = 9 * k;
    const f0 = F[3 * k];
    const f1 = F[3 * k + 1];
    const f2 = F[3 * k + 2];
    Fw[3 * k] = R[o] * f0 + R[o + 1] * f1 + R[o + 2] * f2;
    Fw[3 * k + 1] = R[o + 3] * f0 + R[o + 4] * f1 + R[o + 5] * f2;
    Fw[3 * k + 2] = R[o + 6] * f0 + R[o + 7] * f1 + R[o + 8] * f2;
    const m = Math.sqrt(W[3 * k] * W[3 * k] + W[3 * k + 1] * W[3 * k + 1] + W[3 * k + 2] * W[3 * k + 2]);
    wmag[k] = Number.isFinite(m) ? m : Infinity;
  }
  // rest_mean (preprocess.py:433-439): mean over axis 0 = a left-to-right sum of the selected rows, / n
  const restMean = (from: number, to: number): { m: [number, number, number] | null; n: number } => {
    let s0 = 0;
    let s1 = 0;
    let s2 = 0;
    let cnt = 0;
    for (let k = 0; k < n; k++) {
      if (!(t[k] >= from && t[k] <= to && wmag[k] < STILL_GYRO_DPS)) continue;
      const a = Fw[3 * k];
      const b = Fw[3 * k + 1];
      const c = Fw[3 * k + 2];
      if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c))) continue;
      s0 += a;
      s1 += b;
      s2 += c;
      cnt++;
    }
    return cnt ? { m: [s0 / cnt, s1 / cnt, s2 / cnt], n: cnt } : { m: null, n: 0 };
  };
  const tOn = t[iOn];
  const tEnd = t[iEnd];
  let pre = restMean(tOn - REST_MS, tOn - 1e-9);
  if (pre.n < MIN_REST_N) pre = restMean(tOn - 2 * REST_MS, tOn - 1e-9);
  let gwPre: [number, number, number];
  let preN: number;
  if (pre.n < MIN_REST_N) {
    gwPre = [Fw[3 * iOn], Fw[3 * iOn + 1], Fw[3 * iOn + 2]]; // no fallback on the production path
    preN = 0;
  } else {
    gwPre = pre.m as [number, number, number];
    preN = pre.n;
  }
  let post = restMean(tEnd, tEnd + REST_MS);
  if (post.n < MIN_REST_N) post = restMean(tEnd, tEnd + 2 * REST_MS);
  const gwPost = post.n < MIN_REST_N ? null : (post.m as [number, number, number]);
  const postN = gwPost === null ? 0 : post.n;
  // R[k]^T g0 (einsum 'nji,j->ni')
  const back = (k: number, g0: readonly number[], i: number): number => {
    const o = 9 * k;
    return R[o + i] * g0[0] + R[o + 3 + i] * g0[1] + R[o + 6 + i] * g0[2];
  };
  const g = new Float64Array(3 * n);
  let closureDeg = NaN;
  let method: Gravity['method'] = 'fwd_only';
  if (gwPost !== null) {
    const span = tEnd - tOn;
    for (let k = 0; k < n; k++) {
      let lam: number;
      if (span > 0) {
        lam = (t[k] - tOn) / span;
        lam = lam < 0 ? 0 : lam > 1 ? 1 : lam;
      } else lam = t[k] > tOn ? 1 : 0;
      for (let i = 0; i < 3; i++) g[3 * k + i] = (1.0 - lam) * back(k, gwPre, i) + lam * back(k, gwPost, i);
    }
    closureDeg = angleDeg(gwPre[0], gwPre[1], gwPre[2], gwPost[0], gwPost[1], gwPost[2]);
    method = 'fwd_bwd';
  } else {
    for (let k = 0; k < n; k++) for (let i = 0; i < 3; i++) g[3 * k + i] = back(k, gwPre, i);
  }
  const mPre = Math.sqrt(gwPre[0] * gwPre[0] + gwPre[1] * gwPre[1] + gwPre[2] * gwPre[2]);
  const magnitude = preN > 0 && mPre >= 8.6 && mPre <= 11.0 ? mPre : GRAVITY;
  for (let k = 0; k < n; k++) {
    const a = g[3 * k];
    const b = g[3 * k + 1];
    const c = g[3 * k + 2];
    const nr = Math.sqrt(a * a + b * b + c * c);
    if (nr > 0) {
      g[3 * k] = (a / nr) * magnitude;
      g[3 * k + 1] = (b / nr) * magnitude;
      g[3 * k + 2] = (c / nr) * magnitude;
    } else {
      g[3 * k] = NaN;
      g[3 * k + 1] = NaN;
      g[3 * k + 2] = NaN;
    }
  }
  return { g, preN, postN, closureDeg, method };
}

// ───────────────────────────── canonical (standardize.py:239-284) ─────────────────────────────

/** standardize.canonical on a (cropped) stream; throws RangeError where Python raises ValueError. */
export function canonical(stream: Stream, onsetMs: number, endMs: number): Canonical {
  const t = stream.t;
  const n = t.length;
  if (!(Number.isFinite(onsetMs) && Number.isFinite(endMs)) || endMs <= onsetMs) {
    throw new RangeError(`invalid bounds ${onsetMs}..${endMs}`);
  }
  let iOn = searchsortedLeft(t, onsetMs);
  let iEnd = searchsortedRight(t, endMs) - 1;
  iOn = Math.max(iOn, 0);
  iEnd = Math.min(iEnd, n - 1);
  if (iEnd - iOn < 2) throw new RangeError(`bounds ${onsetMs}..${endMs} ms hold fewer than 3 grid samples`);
  const f = fillGapsRows(t, stream.f).data;
  const w = fillGapsRows(t, stream.gyro).data;
  const gg = gyroGravity(t, f, w, iOn, iEnd);
  const g = gg.g;
  const X = new Float64Array(17 * n);
  const put = (c: CanonicalChannel, i: number, v: number): void => {
    X[CH[c] * n + i] = v;
  };
  for (let i = 0; i < n; i++) {
    const g0 = g[3 * i];
    const g1 = g[3 * i + 1];
    const g2 = g[3 * i + 2];
    const gn = Math.sqrt(g0 * g0 + g1 * g1 + g2 * g2);
    // unit_rows: zero (or NaN) norm -> NaN
    const u0 = gn > 0 ? g0 / gn : NaN;
    const u1 = gn > 0 ? g1 / gn : NaN;
    const u2 = gn > 0 ? g2 / gn : NaN;
    const a0 = f[3 * i] - g0;
    const a1 = f[3 * i + 1] - g1;
    const a2 = f[3 * i + 2] - g2;
    const w0 = w[3 * i];
    const w1 = w[3 * i + 1];
    const w2 = w[3 * i + 2];
    const wv = w0 * u0 + w1 * u1 + w2 * u2;
    const av = a0 * u0 + a1 * u1 + a2 * u2;
    const h0 = a0 - av * u0;
    const h1 = a1 - av * u1;
    const h2 = a2 - av * u2;
    const r0 = w0 - wv * u0;
    const r1 = w1 - wv * u1;
    const r2 = w2 - wv * u2;
    put('acc_x', i, a0);
    put('acc_y', i, a1);
    put('acc_z', i, a2);
    put('gyro_x', i, w0);
    put('gyro_y', i, w1);
    put('gyro_z', i, w2);
    put('up_x', i, u0);
    put('up_y', i, u1);
    put('up_z', i, u2);
    put('gyro_mag', i, Math.sqrt(w0 * w0 + w1 * w1 + w2 * w2));
    put('acc_mag', i, Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2));
    put('acc_vert', i, av);
    put('acc_horiz', i, Math.sqrt(h0 * h0 + h1 * h1 + h2 * h2));
    put('yaw_rate', i, wv);
    put('gyro_horiz', i, Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2));
  }
  // yaw = cumulative_trapezoid(wv, t / 1000, initial=0) - its value at i_on (scipy: (d * (y1 + y0)) / 2.0, cumsum)
  const yw = CH.yaw_rate * n;
  const yc = CH.yaw_cum * n;
  X[yc] = 0.0;
  for (let k = 0; k < n - 1; k++) {
    const d = t[k + 1] / 1000.0 - t[k] / 1000.0;
    X[yc + k + 1] = X[yc + k] + (d * (X[yw + k + 1] + X[yw + k])) / 2.0;
  }
  const yOn = X[yc + iOn];
  for (let k = 0; k < n; k++) X[yc + k] = X[yc + k] - yOn;
  // tilt_from_start = angle between u and u[i_on] (degrees)
  const ux = CH.up_x * n;
  const uy = CH.up_y * n;
  const uz = CH.up_z * n;
  const v0 = X[ux + iOn];
  const v1 = X[uy + iOn];
  const v2 = X[uz + iOn];
  for (let k = 0; k < n; k++) X[CH.tilt_from_start * n + k] = angleDeg(X[ux + k], X[uy + k], X[uz + k], v0, v1, v2);
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
  };
}

// ───────────────────────────── time_normalised (standardize.py:290-301) ─────────────────────────────

/** standardize.time_normalised(can, start, end, 64): np.linspace grid, np.interp per channel, yaw re-zeroed, float32. */
export function timeNormalised(can: Canonical, startMs: number, endMs: number): ApproachWindow {
  const n = N_NORM;
  const step = (endMs - startMs) / (n - 1);
  const grid = new Float64Array(n);
  for (let i = 0; i < n; i++) grid[i] = i * step + startMs; // np.linspace: arange * step + start ...
  grid[n - 1] = endMs; // ... and the last point set to stop
  const data = new Float32Array(17 * n);
  const col = new Float64Array(n);
  for (let c = 0; c < 17; c++) {
    const fp = can.X.subarray(c * can.n, (c + 1) * can.n);
    for (let i = 0; i < n; i++) col[i] = npInterp(grid[i], can.t, fp, can.n);
    if (c === CH.yaw_cum) {
      const y0 = col[0];
      for (let i = 0; i < n; i++) col[i] = col[i] - y0;
    }
    for (let i = 0; i < n; i++) data[c * n + i] = col[i]; // Float32Array stores round to nearest (= astype(float32))
  }
  return { data, onsetMs: startMs, endMs };
}

/** detect's own WindowPipeline (see the ownership note at the top). */
export const detectWindowPipeline: WindowPipeline = { cropStream, canonical, timeNormalised };
