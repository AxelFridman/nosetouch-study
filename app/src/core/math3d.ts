/**
 * Small, dependency-free 3D math used by the simulator, orientation detection and tests.
 *
 * Conventions (W3C DeviceOrientation):
 *  - Earth frame: x = East, y = North, z = Up.
 *  - Device frame: x = right edge, y = top edge, z = out of the screen.
 *  - Device orientation angles (alpha, beta, gamma) are intrinsic Tait-Bryan Z-X'-Y'':
 *      R = Rz(alpha) · Rx(beta) · Ry(gamma)
 *    and R maps DEVICE-frame vectors to EARTH-frame vectors (v_earth = R · v_device).
 *    Ranges: alpha ∈ [0, 360), beta ∈ [-180, 180), gamma ∈ [-90, 90).
 *  - rotationRate (a DIFFERENT convention from the angles): alpha = rate about device X, beta = about
 *    device Y, gamma = about device Z (deg/s; W3C CRD 2025, adopted in 2019 to match browsers —
 *    MDN's "alpha = Z" text is outdated). See types.ts › Euler.
 *  - Mat3 is ROW-MAJOR: [m00, m01, m02, m10, m11, m12, m20, m21, m22].
 *  - Quat is [x, y, z, w] (Hamilton, unit).
 *
 * IMPLEMENTED BY: simulation agent. Signatures are a contract — do not change them.
 */
import type { Euler, Vec3 } from './types';

export type Mat3 = [number, number, number, number, number, number, number, number, number];
export type Quat = [number, number, number, number];

export const DEG = Math.PI / 180;

export function vAdd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function vSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function vScale(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k];
}
export function vDot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function vCross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function vNorm(a: Vec3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
/** Returns a unit vector; returns [0,0,0] for a zero vector. */
export function vNormalize(a: Vec3): Vec3 {
  const n = vNorm(a);
  return n > 0 ? [a[0] / n, a[1] / n, a[2] / n] : [0, 0, 0];
}
/** Angle between two vectors in degrees (0..180). NaN if either is zero. */
export function angleBetweenDeg(a: Vec3, b: Vec3): number {
  if (vNorm(a) === 0 || vNorm(b) === 0) return NaN;
  // atan2(|a×b|, a·b) stays accurate near 0° and 180°, where acos(cos) loses precision.
  return Math.atan2(vNorm(vCross(a, b)), vDot(a, b)) / DEG;
}

export function mIdentity(): Mat3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}
export function mMul(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9) as Mat3;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}
export function mT(a: Mat3): Mat3 {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}
export function mVec(a: Mat3, v: Vec3): Vec3 {
  return [
    a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
    a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
    a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
  ];
}
export function mDet(a: Mat3): number {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}
/** Rotation by `angleRad` about unit `axis` (right-hand rule). The axis is re-normalized defensively; a zero axis gives I. */
export function mAxisAngle(axis: Vec3, angleRad: number): Mat3 {
  const [x, y, z] = vNormalize(axis);
  if (x === 0 && y === 0 && z === 0) return mIdentity();
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  const t = 1 - c;
  // Rodrigues: R = c·I + s·[k]× + t·k·kᵀ
  return [
    c + t * x * x, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, c + t * y * y, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, c + t * z * z,
  ];
}
/** Columns c0, c1, c2 → matrix [c0 c1 c2]. */
export function mFromColumns(c0: Vec3, c1: Vec3, c2: Vec3): Mat3 {
  return [c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]];
}
/** Rotation vector (axis * angle, radians) of a rotation matrix (SO(3) log). Angle ∈ [0, π]. */
export function mLog(a: Mat3): Vec3 {
  // Via the quaternion: robust at 0 and π, where the textbook acos/skew formula breaks down.
  const [x, y, z, w] = quatFromMat(a); // w ≥ 0
  const s = Math.sqrt(x * x + y * y + z * z); // sin(θ/2)
  if (s < 1e-300) return [0, 0, 0];
  const k = (2 * Math.atan2(s, w)) / s;
  return [x * k, y * k, z * k];
}
/** Exponential map: rotation vector (radians) → rotation matrix. */
export function mExp(w: Vec3): Mat3 {
  const angle = vNorm(w);
  if (angle < 1e-12) {
    // first order: I + [w]×  (the second-order term is below double precision here)
    return [1, -w[2], w[1], w[2], 1, -w[0], -w[1], w[0], 1];
  }
  return mAxisAngle(vScale(w, 1 / angle), angle);
}

/** W3C: R = Rz(alpha)·Rx(beta)·Ry(gamma); angles in DEGREES. */
export function matFromEulerZXY(alphaDeg: number, betaDeg: number, gammaDeg: number): Mat3 {
  const cA = Math.cos(alphaDeg * DEG);
  const sA = Math.sin(alphaDeg * DEG);
  const cB = Math.cos(betaDeg * DEG);
  const sB = Math.sin(betaDeg * DEG);
  const cG = Math.cos(gammaDeg * DEG);
  const sG = Math.sin(gammaDeg * DEG);
  // Equation of the W3C DeviceOrientation spec ("rotation matrix" section).
  return [
    cA * cG - sA * sB * sG, -cB * sA, cG * sA * sB + cA * sG,
    cG * sA + cA * sB * sG, cA * cB, sA * sG - cA * cG * sB,
    -cB * sG, sB, cB * cG,
  ];
}

/**
 * Inverse of matFromEulerZXY with W3C ranges (alpha [0,360), beta [-180,180), gamma [-90,90)).
 *
 * Follows the W3C / Chromium decomposition (orientation_util.cc): cases on the sign of
 * m22 = cos(beta)·cos(gamma) (cos(gamma) ≥ 0 in the W3C range, so its sign is the sign of cos(beta));
 * m22 == 0 means gamma = −90° (then the sign of m20 = −cos(beta)·sin(gamma) gives cos(beta)), and
 * m20 == m22 == 0 is gimbal lock (beta = ±90°), where only alpha ± gamma is defined → gamma = 0.
 * beta uses atan2(m21, hypot(m20, m22)) (= asin(m21) exactly in theory) because asin loses
 * precision near ±90°.
 */
export function eulerZXYFromMat(m: Mat3): Euler {
  const [r0, r1, , r3, r4, , r6, r7, r8] = m;
  const sB = Math.min(1, Math.max(-1, r7));
  let a: number;
  let b: number;
  let g: number;
  if (r8 > 0) {
    // cos(beta) > 0 → beta ∈ (−90°, 90°)
    a = Math.atan2(-r1, r4);
    b = Math.atan2(sB, Math.hypot(r6, r8));
    g = Math.atan2(-r6, r8);
  } else if (r8 < 0) {
    // cos(beta) < 0 → beta ∈ [−180°, −90°) ∪ (90°, 180°)
    a = Math.atan2(r1, -r4);
    b = -Math.atan2(sB, Math.hypot(r6, r8));
    b += b >= 0 ? -Math.PI : Math.PI;
    g = Math.atan2(r6, -r8);
  } else if (r6 > 0) {
    // cos(gamma) = 0 (gamma = −90°), cos(beta) > 0
    a = Math.atan2(-r1, r4);
    b = Math.asin(sB);
    g = -Math.PI / 2;
  } else if (r6 < 0) {
    // cos(gamma) = 0 (gamma = −90°), cos(beta) < 0
    a = Math.atan2(r1, -r4);
    b = -Math.asin(sB);
    b += b >= 0 ? -Math.PI : Math.PI;
    g = -Math.PI / 2;
  } else {
    // gimbal lock, cos(beta) = 0: m00 = cos(alpha ± gamma), m10 = sin(alpha ± gamma)
    a = Math.atan2(r3, r0);
    b = r7 > 0 ? Math.PI / 2 : -Math.PI / 2;
    g = 0;
  }
  let alpha = a / DEG;
  let beta = b / DEG;
  let gamma = g / DEG;
  if (gamma >= 90) {
    // atan2 can round to exactly +90° when cos(gamma) is a few ulps above 0. Same rotation:
    // (alpha, beta, gamma) ≡ (alpha + 180, 180 − beta, gamma − 180).
    alpha += 180;
    beta = 180 - beta;
    gamma -= 180;
  }
  alpha = ((alpha % 360) + 360) % 360;
  if (alpha >= 360) alpha -= 360; // (x % 360 + 360) % 360 can round up to 360 for tiny negative x
  if (beta >= 180) beta -= 360;
  if (beta < -180) beta += 360;
  // "+ 0" turns −0 into +0 so serialized/compared values are canonical.
  return { alpha: alpha + 0, beta: beta + 0, gamma: gamma + 0 };
}

/** Unit quaternion [x, y, z, w] of a rotation matrix (Shepperd's method), canonical sign w ≥ 0. */
export function quatFromMat(m: Mat3): Quat {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
  const tr = m00 + m11 + m22;
  let x: number;
  let y: number;
  let z: number;
  let w: number;
  // Branch on the largest of (w, x, y, z) so the division is never by a small number.
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2; // 4w
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(Math.max(0, 1 + m00 - m11 - m22)) * 2; // 4x
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(Math.max(0, 1 + m11 - m00 - m22)) * 2; // 4y
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(Math.max(0, 1 + m22 - m00 - m11)) * 2; // 4z
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const n = Math.sqrt(x * x + y * y + z * z + w * w);
  const k = (w < 0 ? -1 : 1) / n;
  return [x * k + 0, y * k + 0, z * k + 0, w * k + 0];
}
/** Rotation matrix of a quaternion [x, y, z, w] (normalized first, so slightly off-unit input is fine). */
export function matFromQuat(q: Quat): Mat3 {
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n === 0) return mIdentity();
  const x = q[0] / n;
  const y = q[1] / n;
  const z = q[2] / n;
  const w = q[3] / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

/** rotationRate {alpha,beta,gamma} (deg/s) → device-frame angular velocity [x, y, z] = [alpha, beta, gamma] (deg/s, W3C 2025 mapping). */
export function rotationRateToVector(rr: Euler): Vec3 {
  return [rr.alpha, rr.beta, rr.gamma];
}
/** device-frame angular velocity [x, y, z] (deg/s) → rotationRate {alpha: x, beta: y, gamma: z} (W3C 2025 mapping). */
export function vectorToRotationRate(w: Vec3): Euler {
  return { alpha: w[0], beta: w[1], gamma: w[2] };
}
