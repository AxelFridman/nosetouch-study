/**
 * Calibration math (pure, DOM-free). The calibration screen records short sample windows and turns
 * them into a CalibrationResult. Why it exists: platforms disagree on sensor conventions — iOS Safari
 * forwards Core Motion's sign (a face-up phone reads accg z ≈ −9.8, the W3C/Android value is +9.8),
 * and rotationRate's field→axis mapping changed in the spec in 2019. We measure instead of assuming.
 *
 *  poseStats(samples): mean and population sd of accg / acc / rr over the window. still = every
 *    accg axis sd < 0.25 m/s² AND (rr null OR every rr field sd < 3 deg/s) AND n ≥ 20.
 *    (n = number of samples carrying a finite accg vector: a window without accelerometer data must
 *    never count as "still".)
 *  gravityFromFlat(flat): phone flat, screen up. Valid when 8.6 ≤ |accg_mean| ≤ 11.0 and
 *    |z| ≥ 0.9·|accg_mean| → sign = sign(z) (+1 W3C, −1 inverted), magnitude = |accg_mean|; else
 *    nulls plus a note explaining why ("not flat", "moving", "implausible magnitude").
 *    The window must also be still (otherwise "moving", or "too few samples" when n < 20).
 *  uprightCheck(upright, sign): phone upright facing the participant → y·(sign ?? 1) > 7 → 'ok',
 *    < −7 → 'inverted', otherwise 'unclear'; null stats → null.
 *  axisCheck(samples): tilt gesture (flat → tilt sideways ≥ 30° → back). Algorithm:
 *    1. g_i = EMA(accg; τ = 100 ms) (raw sign — irrelevant, see below); g0 = mean of g over the
 *       first 300 ms; i* = argmax angle(g0, g_i); tilt_deg = that angle. tilt < 30° → 'unclear'.
 *    2. Observed device rotation axis n = −normalize(g0 × g_{i*}) (a world-fixed vector seen from
 *       the device rotates opposite to the device; negating both g's for the iOS sign leaves the
 *       cross product unchanged, so this is sign-convention independent).
 *    3. I_f = trapezoidal ∫ rr.f dt (s) from the first sample to i*, for f in alpha, beta, gamma.
 *    4. Candidates in device xyz: w3c = [Iα, Iβ, Iγ]; legacy = [Iβ, Iγ, Iα]; plus their negations.
 *       Expected e = tilt_deg · n. Best = max cosine similarity with e. ratio = |best| / tilt_deg.
 *    5. verdict = best candidate name if similarity ≥ 0.85 AND ratio ∈ [0.6, 1.6] (units
 *       'deg_per_s') or ratio ∈ [0.6, 1.6]/57.2958 (units 'rad_per_s'); else 'unclear'
 *       (units null when neither range matches). rr missing → 'unclear'.
 *    When the tilt is below 30° the verdict is 'unclear' and units stay null; tilt_deg, device_axis,
 *    rr_integral_deg, similarity and magnitude_ratio are still reported as diagnostics (device_axis
 *    and similarity need at least a 1° tilt to mean anything).
 *  engineGravitySign(engine): the sign the browser engine is known to report: webkit (every iOS
 *    browser) −1, blink / gecko +1, 'unknown' / absent → null (no expectation: either sign is accepted).
 *    A flat sign that contradicts it most likely means the phone lay screen-DOWN (|z| ≈ |g| either way).
 *  buildCalibration(...): assembles the CalibrationResult. gyro_bias_dps = flat.rr_mean only when the
 *    flat window was still AND gave a gravity sign (a hand on the phone or a phone still settling
 *    measures motion, not bias); otherwise null. With `engine`, a flat sign that contradicts the
 *    engine expectation is kept but noted (FLAT_SIGN_CONTRADICTS_ENGINE).
 *
 * IMPLEMENTED BY: sensors agent.
 */
import type { AxisCheck, CalibrationResult, DeviceInfo, Euler, MotionSample, PoseStats, Vec3 } from './types';

/** Thresholds of the calibration checks (exported so the calibration screen can explain them). */
export const CALIBRATION_LIMITS = {
  still_accg_sd: 0.25, // m/s², every axis
  still_rr_sd: 3, // deg/s, every field
  still_min_n: 20,
  gravity_min: 8.6, // m/s²
  gravity_max: 11.0,
  flat_z_fraction: 0.9, // |z| ≥ 0.9·|g|  ⇔  screen normal within ~26° of vertical
  upright_min_y: 7, // m/s²: top edge within ~44° of straight up
  axis_ema_tau_ms: 100,
  axis_g0_window_ms: 300,
  axis_min_tilt_deg: 30,
  axis_min_similarity: 0.85,
  axis_ratio_min: 0.6,
  axis_ratio_max: 1.6,
} as const;

const L = CALIBRATION_LIMITS;
/** A rad/s gyro integrates to tilt/57.2958 "degrees": that is how the unit error shows up. */
const RAD_TO_DEG = 180 / Math.PI;
/** Below this tilt the cross product of two gravity directions is noise, not an axis. */
const MIN_AXIS_TILT_DEG = 1;

// ───────────────────────────── small vector helpers ─────────────────────────────
// (local on purpose: math3d.ts belongs to the simulator and calibration must not depend on it)

function isFiniteVec3(v: Vec3 | null | undefined): v is Vec3 {
  return (
    Array.isArray(v) && v.length === 3 && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2])
  );
}

function isFiniteEuler(r: Euler | null | undefined): r is Euler {
  return (
    r !== null &&
    typeof r === 'object' &&
    Number.isFinite(r.alpha) &&
    Number.isFinite(r.beta) &&
    Number.isFinite(r.gamma)
  );
}

function norm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Angle in degrees; atan2 form stays accurate near 0° and 180° (acos does not). */
function angleDeg(a: Vec3, b: Vec3): number {
  return Math.atan2(norm(cross(a, b)), dot(a, b)) * RAD_TO_DEG;
}

function cosine(a: Vec3, b: Vec3): number | null {
  const d = norm(a) * norm(b);
  return d > 0 ? dot(a, b) / d : null;
}

interface MeanSd {
  n: number;
  mean: Vec3;
  sd: Vec3;
}

/** Mean and POPULATION standard deviation per column (two-pass: no cancellation error). */
function meanSd(rows: Vec3[]): MeanSd {
  const n = rows.length;
  if (n === 0) return { n, mean: [0, 0, 0], sd: [0, 0, 0] };
  const mean: Vec3 = [0, 0, 0];
  for (const r of rows) for (let d = 0; d < 3; d++) mean[d] += r[d];
  for (let d = 0; d < 3; d++) mean[d] /= n;
  const sd: Vec3 = [0, 0, 0];
  for (const r of rows) for (let d = 0; d < 3; d++) sd[d] += (r[d] - mean[d]) ** 2;
  for (let d = 0; d < 3; d++) sd[d] = Math.sqrt(sd[d] / n);
  return { n, mean, sd };
}

function toEuler(v: Vec3): Euler {
  return { alpha: v[0], beta: v[1], gamma: v[2] };
}

function fmt(x: number, digits = 2): string {
  return x.toFixed(digits);
}

// ───────────────────────────── pose statistics ─────────────────────────────

export function poseStats(samples: MotionSample[]): PoseStats {
  const accgRows: Vec3[] = [];
  const accRows: Vec3[] = [];
  const rrRows: Vec3[] = [];
  let tMin = Infinity;
  let tMax = -Infinity;
  for (const s of samples) {
    if (Number.isFinite(s.t)) {
      if (s.t < tMin) tMin = s.t;
      if (s.t > tMax) tMax = s.t;
    }
    if (isFiniteVec3(s.accg)) accgRows.push([s.accg[0], s.accg[1], s.accg[2]]);
    if (isFiniteVec3(s.acc)) accRows.push([s.acc[0], s.acc[1], s.acc[2]]);
    if (isFiniteEuler(s.rr)) rrRows.push([s.rr.alpha, s.rr.beta, s.rr.gamma]);
  }
  const accg = meanSd(accgRows);
  const acc = meanSd(accRows);
  const rr = meanSd(rrRows);
  const rrStill = rr.n === 0 || rr.sd.every((sd) => sd < L.still_rr_sd);
  const still = accg.n >= L.still_min_n && accg.sd.every((sd) => sd < L.still_accg_sd) && rrStill;
  return {
    n: accg.n,
    duration_ms: tMax > tMin ? tMax - tMin : 0,
    accg_mean: accg.mean,
    accg_sd: accg.sd,
    acc_mean: acc.n > 0 ? acc.mean : null,
    rr_mean: rr.n > 0 ? toEuler(rr.mean) : null,
    rr_sd: rr.n > 0 ? toEuler(rr.sd) : null,
    still,
  };
}

// ───────────────────────────── gravity sign / upright ─────────────────────────────

export function gravityFromFlat(flat: PoseStats | null): {
  sign: 1 | -1 | null;
  magnitude: number | null;
  notes: string[];
} {
  if (!flat) return { sign: null, magnitude: null, notes: ['flat: not recorded'] };
  if (flat.n === 0 || !isFiniteVec3(flat.accg_mean)) {
    return { sign: null, magnitude: null, notes: ['flat: no accelerometer samples'] };
  }
  const notes: string[] = [];
  const z = flat.accg_mean[2];
  const mag = norm(flat.accg_mean);
  if (!flat.still) {
    if (flat.n < L.still_min_n) {
      notes.push(`flat: too few samples (n = ${flat.n}, need ${L.still_min_n})`);
    } else {
      const accgSd = Math.max(...flat.accg_sd);
      const rrSd = flat.rr_sd ? Math.max(flat.rr_sd.alpha, flat.rr_sd.beta, flat.rr_sd.gamma) : null;
      notes.push(
        `flat: moving (accg sd up to ${fmt(accgSd)} m/s²` +
          (rrSd === null ? ')' : `, rotation rate sd up to ${fmt(rrSd)} deg/s)`),
      );
    }
  }
  if (!(mag >= L.gravity_min && mag <= L.gravity_max)) {
    notes.push(`flat: implausible magnitude ${fmt(mag)} m/s² (expected ${L.gravity_min}–${L.gravity_max})`);
  }
  if (mag > 0 && Math.abs(z) < L.flat_z_fraction * mag) {
    notes.push(`flat: not flat (|z| is ${Math.round((Math.abs(z) / mag) * 100)}% of |g|, need ≥ 90%)`);
  }
  if (notes.length > 0) return { sign: null, magnitude: null, notes };
  return { sign: z > 0 ? 1 : -1, magnitude: mag, notes };
}

/** Calibration note written when the participant continued with a flat sign the engine contradicts. */
export const FLAT_SIGN_CONTRADICTS_ENGINE = 'flat_sign_contradicts_engine';

type Engine = DeviceInfo['engine'] | null | undefined;

/** Accelerometer sign an engine is known to report (WebKit inverts both fields); null = no expectation. */
export function engineGravitySign(engine: Engine): 1 | -1 | null {
  if (engine === 'webkit') return -1;
  if (engine === 'blink' || engine === 'gecko') return 1;
  return null;
}

/** true when a measured flat sign contradicts the engine expectation (typically a screen-down phone). */
export function flatSignContradictsEngine(sign: 1 | -1 | null, engine: Engine): boolean {
  const expected = engineGravitySign(engine);
  return sign !== null && expected !== null && sign !== expected;
}

export function uprightCheck(upright: PoseStats | null, sign: 1 | -1 | null): CalibrationResult['upright_check'] {
  if (!upright) return null;
  const y = upright.accg_mean[1] * (sign ?? 1);
  if (!Number.isFinite(y)) return 'unclear';
  if (y > L.upright_min_y) return 'ok';
  if (y < -L.upright_min_y) return 'inverted';
  return 'unclear';
}

// ───────────────────────────── rotationRate axis check ─────────────────────────────

type AxisVerdict = AxisCheck['verdict'];

export function axisCheck(samples: MotionSample[]): AxisCheck {
  let tMin = Infinity;
  let tMax = -Infinity;
  for (const s of samples) {
    if (!Number.isFinite(s.t)) continue;
    if (s.t < tMin) tMin = s.t;
    if (s.t > tMax) tMax = s.t;
  }
  const result: AxisCheck = {
    n: samples.length,
    duration_ms: tMax > tMin ? tMax - tMin : 0,
    tilt_deg: null,
    device_axis: null,
    rr_integral_deg: null,
    verdict: 'unclear',
    units: null,
    similarity: null,
    magnitude_ratio: null,
  };

  // 1. Low-passed gravity direction. The EMA weight uses the real dt (event spacing jitters).
  const gIdx: number[] = []; // index into `samples` of each filtered value
  const g: Vec3[] = [];
  let ema: Vec3 | null = null;
  let lastT = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (!isFiniteVec3(s.accg) || !Number.isFinite(s.t)) continue;
    if (ema === null) {
      ema = [s.accg[0], s.accg[1], s.accg[2]];
    } else {
      const k = 1 - Math.exp(-Math.max(0, s.t - lastT) / L.axis_ema_tau_ms);
      ema = [ema[0] + k * (s.accg[0] - ema[0]), ema[1] + k * (s.accg[1] - ema[1]), ema[2] + k * (s.accg[2] - ema[2])];
    }
    lastT = s.t;
    gIdx.push(i);
    g.push(ema);
  }
  if (g.length < 2) return result;

  const tFirst = samples[gIdx[0]].t;
  const g0: Vec3 = [0, 0, 0];
  let n0 = 0;
  for (let k = 0; k < g.length && samples[gIdx[k]].t - tFirst <= L.axis_g0_window_ms; k++) {
    for (let d = 0; d < 3; d++) g0[d] += g[k][d];
    n0++;
  }
  for (let d = 0; d < 3; d++) g0[d] /= n0;
  if (norm(g0) === 0) return result;

  let tilt = -1;
  let kStar = 0;
  for (let k = 0; k < g.length; k++) {
    const a = angleDeg(g0, g[k]);
    if (a > tilt) {
      tilt = a;
      kStar = k;
    }
  }
  const iStar = gIdx[kStar];
  result.tilt_deg = tilt;

  // 2. Rotation axis seen by the accelerometer (sign-convention independent: (−a)×(−b) = a×b).
  let axis: Vec3 | null = null;
  if (tilt >= MIN_AXIS_TILT_DEG) {
    const c = cross(g0, g[kStar]);
    const cn = norm(c);
    if (cn > 0) axis = [-c[0] / cn, -c[1] / cn, -c[2] / cn];
  }
  result.device_axis = axis;

  // 3. Trapezoidal integral of every raw rotationRate field from the first sample to i*.
  const integral: Vec3 = [0, 0, 0];
  let prev: { t: number; v: Vec3 } | null = null;
  let nRr = 0;
  for (let i = 0; i <= iStar; i++) {
    const s = samples[i];
    if (!isFiniteEuler(s.rr) || !Number.isFinite(s.t)) continue;
    const v: Vec3 = [s.rr.alpha, s.rr.beta, s.rr.gamma];
    if (prev && s.t > prev.t) {
      const dtS = (s.t - prev.t) / 1000;
      for (let d = 0; d < 3; d++) integral[d] += ((prev.v[d] + v[d]) / 2) * dtS;
    }
    prev = { t: s.t, v };
    nRr++;
  }
  if (nRr < 2) return result; // rr missing → 'unclear'
  const [ia, ib, ig] = integral;
  result.rr_integral_deg = { alpha: ia, beta: ib, gamma: ig };
  if (!axis) return result;

  // 4. Which field→axis mapping explains the observed rotation?
  const candidates: [AxisVerdict, Vec3][] = [
    ['w3c', [ia, ib, ig]],
    ['w3c_inverted', [-ia, -ib, -ig]],
    ['legacy', [ib, ig, ia]],
    ['legacy_inverted', [-ib, -ig, -ia]],
  ];
  const expected: Vec3 = [axis[0] * tilt, axis[1] * tilt, axis[2] * tilt];
  let best: AxisVerdict = 'unclear';
  let bestSim = -Infinity;
  for (const [name, vec] of candidates) {
    const sim = cosine(vec, expected);
    if (sim !== null && sim > bestSim) {
      bestSim = sim;
      best = name;
    }
  }
  if (best === 'unclear') return result; // zero integral: nothing to compare
  const ratio = norm(integral) / tilt; // identical for every candidate (they are permutations)
  result.similarity = bestSim;
  result.magnitude_ratio = ratio;
  if (tilt < L.axis_min_tilt_deg) return result;

  // 5. Verdict and units.
  if (ratio >= L.axis_ratio_min && ratio <= L.axis_ratio_max) result.units = 'deg_per_s';
  else if (ratio >= L.axis_ratio_min / RAD_TO_DEG && ratio <= L.axis_ratio_max / RAD_TO_DEG) result.units = 'rad_per_s';
  if (bestSim >= L.axis_min_similarity && result.units !== null) result.verdict = best;
  return result;
}

// ───────────────────────────── assembly ─────────────────────────────

export function buildCalibration(parts: {
  performedAt: string;
  flat: PoseStats | null;
  upright: PoseStats | null;
  axis: AxisCheck | null;
  notes?: string[];
  /** device.engine of this browser (null in simulation: the simulator's sign is not the engine's) */
  engine?: Engine;
}): CalibrationResult {
  const gravity = gravityFromFlat(parts.flat);
  const upright_check = uprightCheck(parts.upright, gravity.sign);
  const notes = [...gravity.notes];
  if (flatSignContradictsEngine(gravity.sign, parts.engine)) notes.push(FLAT_SIGN_CONTRADICTS_ENGINE);
  if (parts.upright && upright_check !== 'ok') {
    notes.push(`upright: ${upright_check} (accg y = ${fmt(parts.upright.accg_mean[1])} m/s², gravity sign ${gravity.sign ?? 'unknown'})`);
  }
  const axis = parts.axis;
  if (axis && axis.verdict !== 'w3c') {
    const tilt = axis.tilt_deg === null ? 'n/a' : `${fmt(axis.tilt_deg, 1)}°`;
    const sim = axis.similarity === null ? 'n/a' : fmt(axis.similarity);
    notes.push(`axis check: ${axis.verdict} (tilt ${tilt}, similarity ${sim}, units ${axis.units ?? 'unknown'})`);
  } else if (axis && axis.units === 'rad_per_s') {
    notes.push('axis check: rotationRate is in rad/s, not deg/s');
  }
  if (parts.notes) notes.push(...parts.notes);
  // Only a window that passed every flat check counts as rest: a moving one measured the hand (or a
  // phone still settling), not the gyroscope's bias, and norm_v1 would subtract it from every trial.
  const biasUsable = parts.flat !== null && parts.flat.still && gravity.sign !== null;
  return {
    performed_at: parts.performedAt,
    flat: parts.flat,
    upright: parts.upright,
    gravity_sign: gravity.sign,
    gravity_magnitude: gravity.magnitude,
    upright_check,
    gyro_bias_dps: biasUsable ? (parts.flat?.rr_mean ?? null) : null,
    notes,
    axis_check: axis,
  };
}
