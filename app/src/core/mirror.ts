/**
 * Left↔right mirroring (reflection of the body across its sagittal plane).
 *
 * Derivation (see docs/05_MODELING_PLAN.md): with R(t) the device→body rotation, a body-frame
 * reflection M = diag(-1,1,1) and the phone identified with its mirror image through the
 * device-frame reflection S = diag(-1,1,1), the mirrored phone pose is R' = M·R·S and the readings
 * transform, FOR ANY GRIP ORIENTATION, as:
 *     acceleration (polar vector):         a' = S·a      → (-ax,  ay,  az)
 *     angular velocity (pseudo-vector):    ω' = -S·ω     → ( ωx, -ωy, -ωz)
 * i.e. with the W3C 2025 rotationRate mapping (alpha = x, beta = y, gamma = z): alpha (x)
 * unchanged, beta (y) negated, gamma (z) negated. Data from a device whose calibration
 * axis_check.verdict is not 'w3c' must be mapped to x/y/z BEFORE mirroring (docs/02_DATA_SCHEMA.md).
 * The instructed orientation maps through ORIENTATIONS[o].mirror (top_left ↔ top_right).
 * Orientation angles (world-referenced) transform as R' = Mw·R·S with Mw a reflection across ANY
 * vertical plane of the earth frame (heading is arbitrary) — use Mw = diag(-1, 1, 1) in earth axes.
 * (For generic angles this is (alpha, beta, gamma) → (360 − alpha, beta, −gamma); going through the
 * matrix also handles gamma = −90° and gimbal lock.) Quaternions: (x, y, z, w) → (x, −y, −z, w).
 * Compass headings (clockwise from North) → 360 − heading. Screen-orientation angles → 360 − angle
 * (landscape-primary ↔ landscape-secondary). Magnetometer readings (device frame) of the earth's
 * field, which lies in the North–South vertical plane that Mw reflects across (the same choice as the
 * compass): m' = S·m → (−mx, my, mz), like acceleration. The schema 1.2 absolute streams
 * (orientation_abs, quaternion_abs) mirror like their relative twins.
 *
 * Mirroring is an involution: mirroring twice returns the original values (up to float rounding
 * for the Euler angles, exactly for everything else).
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { ORIENTATIONS } from './constants';
import { eulerZXYFromMat, matFromEulerZXY, type Mat3 } from './math3d';
import type {
  Hand,
  MotionSample,
  OrientationSample,
  OrientationStream,
  PhoneRotation,
  QuaternionStream,
  TrialCondition,
  TrialRecord,
  Vec3,
} from './types';

function otherHand(h: Hand): Hand {
  return h === 'left' ? 'right' : 'left';
}

/** x → −x of a device- or body-frame polar vector. */
function mirrorPolar(v: Vec3): Vec3 {
  return [-v[0], v[1], v[2]];
}

function neg(v: number | null): number | null {
  return v === null ? null : -v;
}

/** Heading in degrees (clockwise from North) reflected across the North–South plane, in [0, 360). */
function mirrorHeading(deg: number): number {
  const r = (((360 - deg) % 360) + 360) % 360;
  return r + 0; // no −0
}

/** M·R·M with M = diag(−1, 1, 1): negate row 0 and column 0 (m00 is negated twice). */
function reflectX(m: Mat3): Mat3 {
  return [m[0], -m[1], -m[2], -m[3], m[4], m[5], -m[6], m[7], m[8]];
}

/** Mirrors one Euler triple; null if any angle is missing or non-finite (no partial mirroring). */
function mirrorEuler(
  alpha: number | null,
  beta: number | null,
  gamma: number | null,
): { alpha: number | null; beta: number | null; gamma: number | null } {
  if (
    alpha === null || beta === null || gamma === null ||
    !Number.isFinite(alpha) || !Number.isFinite(beta) || !Number.isFinite(gamma)
  ) {
    return { alpha: null, beta: null, gamma: null };
  }
  // Mw·R·S with Mw = S = diag(−1,1,1) is numerically the same matrix operation as M·R·M.
  const e = eulerZXYFromMat(reflectX(matFromEulerZXY(alpha, beta, gamma)));
  return { alpha: e.alpha, beta: e.beta, gamma: e.gamma };
}

function mirrorRotation(r: PhoneRotation | 'unknown'): PhoneRotation | 'unknown' {
  if (r === 'top_left') return 'top_right';
  if (r === 'top_right') return 'top_left';
  return r;
}

function mirrorScreenType(type: string | null): string | null {
  if (type === 'landscape-primary') return 'landscape-secondary';
  if (type === 'landscape-secondary') return 'landscape-primary';
  return type;
}

export function mirrorCondition(c: TrialCondition): TrialCondition {
  return { ...c, hand: otherHand(c.hand), orientation: ORIENTATIONS[c.orientation].mirror };
}

export function mirrorMotionSample(s: MotionSample): MotionSample {
  return {
    t: s.t,
    ts: s.ts,
    acc: s.acc ? mirrorPolar(s.acc) : null,
    accg: s.accg ? mirrorPolar(s.accg) : null,
    // ω' = −S·ω: x (alpha) kept, y (beta) and z (gamma) negated
    rr: s.rr ? { alpha: s.rr.alpha, beta: -s.rr.beta, gamma: -s.rr.gamma } : null,
    interval: s.interval,
  };
}

/** Mirrors W3C Euler angles: R' = Mw·R·S, re-expressed with eulerZXYFromMat. Nulls stay null. */
export function mirrorOrientationSample(s: OrientationSample): OrientationSample {
  const e = mirrorEuler(s.alpha, s.beta, s.gamma);
  const out: OrientationSample = {
    t: s.t,
    ts: s.ts,
    alpha: e.alpha,
    beta: e.beta,
    gamma: e.gamma,
    absolute: s.absolute,
    compass: s.compass === null ? null : mirrorHeading(s.compass),
  };
  if (s.compass_accuracy !== undefined) out.compass_accuracy = s.compass_accuracy; // an uncertainty, not a direction
  return out;
}

/** In place: Euler angles and compass headings of an orientation stream (compass_accuracy is kept). */
function mirrorOrientationStream(o: OrientationStream | null | undefined): void {
  if (!o) return;
  for (let i = 0; i < o.t.length; i++) {
    const e = mirrorEuler(o.alpha[i], o.beta[i], o.gamma[i]);
    o.alpha[i] = e.alpha;
    o.beta[i] = e.beta;
    o.gamma[i] = e.gamma;
  }
  o.compass = o.compass.map((c) => (c === null ? null : mirrorHeading(c)));
}

/** In place: (x, y, z, w) → (x, −y, −z, w). */
function mirrorQuaternionStream(q: QuaternionStream | null | undefined): void {
  if (!q) return;
  q.y = q.y.map((v) => -v);
  q.z = q.z.map((v) => -v);
}

/**
 * Returns a deep-copied trial whose streams, condition and detected rotation are mirrored.
 * Also mirrored, for consistency: segmentation.baseline.gravity (device-frame vector), the screen
 * orientation (angle → 360 − angle, landscape-primary ↔ landscape-secondary) and the schema 1.2
 * streams (orientation_abs, quaternion_abs, magnetometer, screen_events).
 * NOT mirrored: touches (x/y in px; the screen width is not stored, and the contact details follow
 * x/y), ids, timing, outcome and quality.
 */
export function mirrorTrial(trial: TrialRecord): TrialRecord {
  const out = structuredClone(trial);
  out.condition = mirrorCondition(trial.condition);

  const m = out.streams.motion;
  m.acc_x = m.acc_x.map(neg);
  m.accg_x = m.accg_x.map(neg);
  m.rr_beta = m.rr_beta.map(neg); // y
  m.rr_gamma = m.rr_gamma.map(neg); // z

  mirrorOrientationStream(out.streams.orientation);
  mirrorOrientationStream(out.streams.orientation_abs);
  mirrorQuaternionStream(out.streams.quaternion);
  mirrorQuaternionStream(out.streams.quaternion_abs);

  const mag = out.streams.magnetometer;
  if (mag) mag.x = mag.x.map(neg);

  const screens = out.streams.screen_events;
  if (screens) {
    for (const s of screens) {
      s.angle = s.angle === null ? null : mirrorHeading(s.angle);
      s.type = mirrorScreenType(s.type);
    }
  }

  out.detected.rotation_start = mirrorRotation(trial.detected.rotation_start);
  out.detected.rotation_end = mirrorRotation(trial.detected.rotation_end);

  const g = out.segmentation.baseline?.gravity;
  if (out.segmentation.baseline && g) out.segmentation.baseline.gravity = mirrorPolar(g);

  out.screen.angle_start = out.screen.angle_start === null ? null : mirrorHeading(out.screen.angle_start);
  out.screen.angle_end = out.screen.angle_end === null ? null : mirrorHeading(out.screen.angle_end);
  out.screen.type_start = mirrorScreenType(out.screen.type_start);
  out.screen.type_end = mirrorScreenType(out.screen.type_end);
  return out;
}
