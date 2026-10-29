/**
 * gravity_v1: integrate_gyro and gyro_gravity (motionlab/preprocess.py:376-418 and :465-544), PORT_SPEC 2.5.
 *
 * Bit-level notes (checked against numpy 2.4 on this machine):
 *  - integrate_gyro is plain Python float arithmetic (same expressions here); math.sin / math.cos come from the
 *    C runtime, Math.sin / Math.cos from V8, which may differ by one ulp in rare cases;
 *  - Fw = einsum('nij,nj->ni', R, F) sums (R[i][0] F0 + R[i][2] F2) + R[i][1] F1 (numpy's SIMD pairing; 0
 *    mismatches against numpy for n = 3..1000), while einsum('nji,j->ni', R, g) sums left to right;
 *  - the rest means f[sel].mean(axis=0) add the rows in order and divide by the count;
 *  - |g_pre| is np.linalg.norm of one vector, i.e. BLAS ddot: see numeric.norm3Dot.
 */
import { CAL_MAG_MAX, CAL_MAG_MIN, DEG2RAD, GRAVITY, MAG_MAX, MAG_MIN, MIN_REST_N, REST_MS, STILL_GYRO_DPS } from './constants';
import { angleDeg3, norm3, norm3Dot } from './numeric';

/**
 * integrate_gyro(t_ms, w_dps): R[k] (row-major 3x3 at 9k) maps device-frame vectors at sample k into the device
 * frame of sample 0, from the midpoint rate of each step; steps with a non-finite rate or dt keep the orientation
 * and are counted in `missing`.
 */
export function integrateGyro(tMs: ArrayLike<number>, wDps: ArrayLike<number>, n: number): { R: Float64Array; missing: number } {
  const qs = new Float64Array(4 * n);
  let x = 0.0;
  let y = 0.0;
  let z = 0.0;
  let w = 1.0;
  if (n > 0) qs[3] = 1.0;
  let missing = 0;
  for (let k = 0; k < n - 1; k++) {
    const dt = tMs[k + 1] / 1000.0 - tMs[k] / 1000.0; // t = t_ms / 1000.0 first, then t[k + 1] - t[k]
    const a0 = wDps[3 * k] * DEG2RAD;
    const a1 = wDps[3 * k + 1] * DEG2RAD;
    const a2 = wDps[3 * k + 2] * DEG2RAD;
    const b0 = wDps[3 * k + 3] * DEG2RAD;
    const b1 = wDps[3 * k + 4] * DEG2RAD;
    const b2 = wDps[3 * k + 5] * DEG2RAD;
    const o = 4 * (k + 1);
    if (
      !(Number.isFinite(a0) && Number.isFinite(a1) && Number.isFinite(a2)) ||
      !(Number.isFinite(b0) && Number.isFinite(b1) && Number.isFinite(b2)) ||
      !Number.isFinite(dt)
    ) {
      missing += 1;
      qs[o] = x;
      qs[o + 1] = y;
      qs[o + 2] = z;
      qs[o + 3] = w;
      continue;
    }
    const ox = (a0 + b0) * 0.5 * dt;
    const oy = (a1 + b1) * 0.5 * dt;
    const oz = (a2 + b2) * 0.5 * dt;
    const th = Math.sqrt(ox * ox + oy * oy + oz * oz);
    if (th < 1e-15) {
      qs[o] = x;
      qs[o + 1] = y;
      qs[o + 2] = z;
      qs[o + 3] = w;
      continue;
    }
    const h = 0.5 * th;
    const s = Math.sin(h) / th;
    const dx = ox * s;
    const dy = oy * s;
    const dz = oz * s;
    const dw = Math.cos(h);
    // q <- q (x) dq (body-frame increment), preprocess.py:409-414
    const nx = w * dx + x * dw + y * dz - z * dy;
    const ny = w * dy - x * dz + y * dw + z * dx;
    const nz = w * dz + x * dy - y * dx + z * dw;
    const nw = w * dw - x * dx - y * dy - z * dz;
    const nq = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
    x = nx / nq;
    y = ny / nq;
    z = nz / nq;
    w = nw / nq;
    qs[o] = x;
    qs[o + 1] = y;
    qs[o + 2] = z;
    qs[o + 3] = w;
  }
  return { R: quaternionsToMatrices(qs, n), missing };
}

/** quaternion_to_matrix (preprocess.py:304-319): normalise again, then the standard [x, y, z, w] matrix. */
export function quaternionsToMatrices(qs: ArrayLike<number>, n: number): Float64Array {
  const R = new Float64Array(9 * n);
  for (let k = 0; k < n; k++) {
    const q0 = qs[4 * k];
    const q1 = qs[4 * k + 1];
    const q2 = qs[4 * k + 2];
    const q3 = qs[4 * k + 3];
    const nrm = Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    const x = q0 / nrm;
    const y = q1 / nrm;
    const z = q2 / nrm;
    const w = q3 / nrm;
    const o = 9 * k;
    R[o] = 1 - 2 * (y * y + z * z);
    R[o + 1] = 2 * (x * y - z * w);
    R[o + 2] = 2 * (x * z + y * w);
    R[o + 3] = 2 * (x * y + z * w);
    R[o + 4] = 1 - 2 * (x * x + z * z);
    R[o + 5] = 2 * (y * z - x * w);
    R[o + 6] = 2 * (x * z - y * w);
    R[o + 7] = 2 * (y * z + x * w);
    R[o + 8] = 1 - 2 * (x * x + y * y);
  }
  return R;
}

/** Result of gyroGravity (preprocess.GyroGravity, the fields canonical() reads). */
export interface GyroGravity {
  /** (n, 3) row-major gravity reaction ("up", W3C sign) rescaled to `magnitude` */
  readonly g: Float64Array;
  readonly preN: number;
  readonly postN: number;
  readonly closureDeg: number;
  readonly magnitude: number;
  readonly method: 'fwd_bwd' | 'fwd_only';
  readonly gyroMissingSteps: number;
}

export interface GyroGravityOptions {
  readonly stillDps?: number;
  readonly restMs?: number;
  readonly minRestN?: number;
  /** a calibration |g| (null in production: m4_detect.as_stream drops it) */
  readonly magnitude?: number | null;
  /** the online baseline gravity (null in production) */
  readonly gPreFallback?: readonly number[] | null;
}

/**
 * rest_mean (preprocess.py:433-439) over Fw: the mean of the rows with t_from <= t <= t_to, |w| < still and all
 * three components finite; rows are added in order (numpy's axis-0 sum of a C-ordered array) then divided.
 */
function restMean(
  t: ArrayLike<number>,
  Fw: Float64Array,
  wmag: Float64Array,
  n: number,
  tFrom: number,
  tTo: number,
  still: number,
): { mean: [number, number, number] | null; count: number } {
  let s0 = 0.0;
  let s1 = 0.0;
  let s2 = 0.0;
  let count = 0;
  for (let k = 0; k < n; k++) {
    if (!(t[k] >= tFrom && t[k] <= tTo && wmag[k] < still)) continue;
    const f0 = Fw[3 * k];
    const f1 = Fw[3 * k + 1];
    const f2 = Fw[3 * k + 2];
    if (!(Number.isFinite(f0) && Number.isFinite(f1) && Number.isFinite(f2))) continue;
    s0 += f0;
    s1 += f1;
    s2 += f2;
    count++;
  }
  return count ? { mean: [s0 / count, s1 / count, s2 / count], count } : { mean: null, count: 0 };
}

/**
 * gyro_gravity(t, f, w, i_on, i_end) (preprocess.py:465-544): the rest specific force before the onset and after
 * the end, both averaged in the device frame of sample 0, propagated with the gyroscope (g(k) = R(k)^T g0) and
 * blended linearly in time between the bounds; rescaled to |g_pre| when plausible, else 9.80665.
 * `f` and `w` are (n, 3) row-major and gap-filled (canonical runs fill_gaps_linear first).
 */
export function gyroGravity(
  t: ArrayLike<number>,
  f: Float64Array,
  w: Float64Array,
  n: number,
  iOn: number,
  iEnd: number,
  opts: GyroGravityOptions = {},
): GyroGravity {
  if (!(0 <= iOn && iOn <= iEnd && iEnd < n)) throw new RangeError(`invalid segment ${iOn}..${iEnd} for ${n} samples`);
  const still = opts.stillDps ?? STILL_GYRO_DPS;
  const restMs = opts.restMs ?? REST_MS;
  const minRestN = opts.minRestN ?? MIN_REST_N;
  const { R, missing } = integrateGyro(t, w, n);
  // Fw = einsum('nij,nj->ni', R, F): (R_i0 F0 + R_i2 F2) + R_i1 F1
  const Fw = new Float64Array(3 * n);
  const wmag = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const o = 9 * k;
    const f0 = f[3 * k];
    const f1 = f[3 * k + 1];
    const f2 = f[3 * k + 2];
    for (let i = 0; i < 3; i++) Fw[3 * k + i] = R[o + 3 * i] * f0 + R[o + 3 * i + 2] * f2 + R[o + 3 * i + 1] * f1;
    const m = norm3(w[3 * k], w[3 * k + 1], w[3 * k + 2]);
    wmag[k] = Number.isFinite(m) ? m : Infinity;
  }
  const tOn = t[iOn];
  const tEnd = t[iEnd];
  let pre = restMean(t, Fw, wmag, n, tOn - restMs, tOn - 1e-9, still);
  if (pre.count < minRestN) pre = restMean(t, Fw, wmag, n, tOn - 2 * restMs, tOn - 1e-9, still);
  let gwPre: [number, number, number];
  let preN = pre.count;
  if (pre.count < minRestN) {
    const fb = opts.gPreFallback;
    const o = 9 * iOn;
    if (fb && fb.length === 3 && fb.every((v) => Number.isFinite(v))) {
      // R[i_on] @ fb (numpy matmul; the BLAS order may differ by one ulp - never used in production)
      gwPre = [0, 1, 2].map((i) => R[o + 3 * i] * fb[0] + R[o + 3 * i + 1] * fb[1] + R[o + 3 * i + 2] * fb[2]) as [number, number, number];
    } else {
      gwPre = [Fw[3 * iOn], Fw[3 * iOn + 1], Fw[3 * iOn + 2]];
    }
    preN = 0;
  } else {
    gwPre = pre.mean as [number, number, number];
  }
  let post = restMean(t, Fw, wmag, n, tEnd, tEnd + restMs, still);
  if (post.count < minRestN) post = restMean(t, Fw, wmag, n, tEnd, tEnd + 2 * restMs, still);
  const gwPost = post.count < minRestN ? null : post.mean;
  const postN = gwPost ? post.count : 0;

  const g = new Float64Array(3 * n);
  let closureDeg = NaN;
  let method: GyroGravity['method'] = 'fwd_only';
  // g_fwd[k] = R[k]^T gw_pre (einsum('nji,j->ni'), left to right)
  const span = tEnd - tOn;
  for (let k = 0; k < n; k++) {
    const o = 9 * k;
    for (let i = 0; i < 3; i++) {
      const gf = R[o + i] * gwPre[0] + R[o + 3 + i] * gwPre[1] + R[o + 6 + i] * gwPre[2];
      if (gwPost) {
        const gb = R[o + i] * gwPost[0] + R[o + 3 + i] * gwPost[1] + R[o + 6 + i] * gwPost[2];
        // lam = clip((t - t_on) / span, 0, 1) (span > 0: canonical guarantees i_end - i_on >= 2)
        const lam = span > 0 ? Math.min(Math.max((t[k] - tOn) / span, 0.0), 1.0) : t[k] > tOn ? 1.0 : 0.0;
        g[3 * k + i] = (1.0 - lam) * gf + lam * gb;
      } else {
        g[3 * k + i] = gf;
      }
    }
  }
  if (gwPost) {
    closureDeg = angleDeg3(gwPre[0], gwPre[1], gwPre[2], gwPost[0], gwPost[1], gwPost[2]);
    method = 'fwd_bwd';
  }
  let magnitude = opts.magnitude ?? null;
  if (magnitude === null || !(CAL_MAG_MIN <= magnitude && magnitude <= CAL_MAG_MAX)) {
    const mPre = norm3Dot(gwPre[0], gwPre[1], gwPre[2]);
    magnitude = preN > 0 && MAG_MIN <= mPre && mPre <= MAG_MAX ? mPre : GRAVITY;
  }
  for (let k = 0; k < n; k++) {
    const o = 3 * k;
    const nrm = norm3(g[o], g[o + 1], g[o + 2]);
    for (let i = 0; i < 3; i++) g[o + i] = (nrm > 0 ? g[o + i] / nrm : NaN) * magnitude;
  }
  return { g, preN, postN, closureDeg, magnitude, method, gyroMissingSteps: missing };
}
