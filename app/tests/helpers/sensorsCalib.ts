/**
 * Synthetic MotionSample windows for the calibration tests: a phone at rest in a given pose, and
 * the tilt gesture used by axisCheck. Self-contained math (math3d.ts belongs to the simulator).
 *
 * Physics of the tilt gesture (the derivation the tests rely on):
 *  - Let R(t) map DEVICE coordinates to WORLD coordinates (v_world = R·v_device). The phone starts
 *    in pose R0 and turns by θ(t) about a unit axis n that is FIXED IN THE DEVICE FRAME:
 *        R(t) = R0 · Rot(n, θ(t))                         (right-multiplication = body-fixed axis)
 *  - Angular velocity in the device frame (what a gyroscope measures): from Rᵀ·dR/dt =
 *    Rot(n,θ)ᵀ·R0ᵀ·R0·θ'·Rot(n,θ)·[n]× = θ'·[n]×  →  ω_device(t) = θ'(t)·n.
 *  - A world-fixed vector u (here "up") seen from the device: u_d(t) = R(t)ᵀ·u = Rot(n,θ)ᵀ·R0ᵀ·u
 *        = Rot(n, −θ(t)) · u_d(0)
 *    i.e. it turns by −θ about n: OPPOSITE to the device. That is why axisCheck negates g0 × g*.
 *  - A resting accelerometer measures the reaction to gravity: W3C/Android accg = +g·u_d (face-up
 *    phone: z = +9.81); iOS WebKit forwards Core Motion unnegated: accg = −g·u_d (z = −9.81).
 *  - rotationRate field encodings of ω_device = (ωx, ωy, ωz):
 *        w3c   : alpha = ωx, beta = ωy, gamma = ωz      (W3C CRD 2025, all current engines)
 *        legacy: alpha = ωz, beta = ωx, gamma = ωy      (pre-2019 spec text, MDN's "alpha = Z")
 */
import type { Euler, MotionSample, Vec3 } from '../../src/core/types';

export const G = 9.80665;
const DEG = Math.PI / 180;

/** mulberry32 — deterministic noise without depending on other modules. */
export function seededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller. */
export function gaussian(rng: () => number): () => number {
  return () => {
    const u = Math.max(rng(), 1e-12);
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

export function norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function normalize(v: Vec3): Vec3 {
  const n = norm(v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Rodrigues: v rotated by angleRad about the unit axis k (right-hand rule). */
export function rotate(v: Vec3, k: Vec3, angleRad: number): Vec3 {
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  const kxv = cross(k, v);
  const kv = dot(k, v) * (1 - c);
  return [v[0] * c + kxv[0] * s + k[0] * kv, v[1] * c + kxv[1] * s + k[1] * kv, v[2] * c + kxv[2] * s + k[2] * kv];
}

export type RrEncoding = 'w3c' | 'legacy';

/** Device-frame angular velocity (deg/s) → rotationRate fields in the given encoding/sign/units. */
export function encodeRotationRate(w: Vec3, encoding: RrEncoding, sign: 1 | -1, units: 'deg' | 'rad'): Euler {
  const k = sign * (units === 'rad' ? DEG : 1);
  return encoding === 'w3c'
    ? { alpha: k * w[0], beta: k * w[1], gamma: k * w[2] }
    : { alpha: k * w[2], beta: k * w[0], gamma: k * w[1] };
}

export interface StillWindowSpec {
  /** device-frame direction of "up" (normalized here): [0,0,1] flat screen-up, [0,1,0] upright */
  up: Vec3;
  /** +1 W3C/Android, −1 iOS WebKit */
  accgSign?: 1 | -1;
  n?: number;
  hz?: number;
  magnitude?: number;
  accgNoise?: number; // m/s² sd
  rrNoise?: number; // deg/s sd
  rrBias?: Euler; // deg/s
  withAcc?: boolean;
  withRr?: boolean;
  /** amplitude (m/s²) of a 3 Hz shake added to every accg axis: a phone that is not at rest */
  shake?: number;
  seed?: number;
  t0?: number;
}

export function stillWindow(spec: StillWindowSpec): MotionSample[] {
  const n = spec.n ?? 60;
  const dt = 1000 / (spec.hz ?? 60);
  const sign = spec.accgSign ?? 1;
  const mag = spec.magnitude ?? G;
  const up = normalize(spec.up);
  const randn = gaussian(seededRng(spec.seed ?? 7));
  const an = spec.accgNoise ?? 0.03;
  const rn = spec.rrNoise ?? 0.2;
  const bias = spec.rrBias ?? { alpha: 0.4, beta: -0.3, gamma: 0.2 };
  const shake = spec.shake ?? 0;
  const out: MotionSample[] = [];
  for (let i = 0; i < n; i++) {
    const t = (spec.t0 ?? 1000) + i * dt;
    const wobble = shake * Math.sin(2 * Math.PI * 3 * (i * dt) / 1000);
    const accg: Vec3 = [
      sign * mag * up[0] + an * randn() + wobble,
      sign * mag * up[1] + an * randn() + wobble,
      sign * mag * up[2] + an * randn() + wobble,
    ];
    out.push({
      t,
      ts: t,
      acc: spec.withAcc === false ? null : [an * randn(), an * randn(), an * randn()],
      accg,
      rr:
        spec.withRr === false
          ? null
          : { alpha: bias.alpha + rn * randn(), beta: bias.beta + rn * randn(), gamma: bias.gamma + rn * randn() },
      interval: 16,
    });
  }
  return out;
}

export interface TiltSpec {
  /** device-frame rotation axis (normalized here) */
  axis: Vec3;
  tiltDeg: number;
  /** initial device-frame "up" (normalized here); default [0,0,1] = flat, screen up */
  up0?: Vec3;
  rrEncoding?: RrEncoding;
  rrSign?: 1 | -1;
  rrUnits?: 'deg' | 'rad';
  accgSign?: 1 | -1;
  hz?: number;
  stillMs?: number;
  rampMs?: number;
  holdMs?: number;
  tailMs?: number;
  accgNoise?: number; // m/s² sd
  rrNoise?: number; // deg/s sd
  rrBias?: Vec3; // device-frame gyro bias, deg/s
  seed?: number;
  t0?: number;
  /** rotationRate null on every sample (platform without a gyroscope) */
  noRr?: boolean;
}

/**
 * Tilt gesture: still → raised-cosine ramp to tiltDeg → hold → ramp back → still.
 * θ(t) = Θ·(1 − cos πs)/2 on the way out (s = (t − t_a)/T), θ'(t) = Θ·π/(2T)·sin πs.
 */
export function tiltGesture(spec: TiltSpec): MotionSample[] {
  const n = normalize(spec.axis);
  const up0 = normalize(spec.up0 ?? [0, 0, 1]);
  const hz = spec.hz ?? 60;
  const dt = 1000 / hz;
  const stillMs = spec.stillMs ?? 400;
  const rampMs = spec.rampMs ?? 500;
  const holdMs = spec.holdMs ?? 400;
  const tailMs = spec.tailMs ?? 300;
  const total = stillMs + 2 * rampMs + holdMs + tailMs;
  const Theta = spec.tiltDeg;
  const accgSign = spec.accgSign ?? 1;
  const randn = gaussian(seededRng(spec.seed ?? 11));
  const an = spec.accgNoise ?? 0.03;
  const rn = spec.rrNoise ?? 0.2;
  const bias = spec.rrBias ?? [0.4, -0.3, 0.2];

  // θ (deg) and θ' (deg/s) at time τ (ms since the start of the window)
  const profile = (tau: number): [number, number] => {
    const upStart = stillMs;
    const downStart = stillMs + rampMs + holdMs;
    const T = rampMs / 1000;
    if (tau < upStart) return [0, 0];
    if (tau < upStart + rampMs) {
      const s = (tau - upStart) / rampMs;
      return [(Theta * (1 - Math.cos(Math.PI * s))) / 2, ((Theta * Math.PI) / (2 * T)) * Math.sin(Math.PI * s)];
    }
    if (tau < downStart) return [Theta, 0];
    if (tau < downStart + rampMs) {
      const s = (tau - downStart) / rampMs;
      return [(Theta * (1 + Math.cos(Math.PI * s))) / 2, -((Theta * Math.PI) / (2 * T)) * Math.sin(Math.PI * s)];
    }
    return [0, 0];
  };

  const out: MotionSample[] = [];
  for (let i = 0; i * dt <= total; i++) {
    const tau = i * dt;
    const [thetaDeg, rateDps] = profile(tau);
    const upD = rotate(up0, n, -thetaDeg * DEG); // world "up" turns OPPOSITE to the device
    const accg: Vec3 = [
      accgSign * G * upD[0] + an * randn(),
      accgSign * G * upD[1] + an * randn(),
      accgSign * G * upD[2] + an * randn(),
    ];
    const w: Vec3 = [
      rateDps * n[0] + bias[0] + rn * randn(),
      rateDps * n[1] + bias[1] + rn * randn(),
      rateDps * n[2] + bias[2] + rn * randn(),
    ];
    const t = (spec.t0 ?? 5000) + tau;
    out.push({
      t,
      ts: t,
      acc: [an * randn(), an * randn(), an * randn()],
      accg,
      rr: spec.noRr ? null : encodeRotationRate(w, spec.rrEncoding ?? 'w3c', spec.rrSign ?? 1, spec.rrUnits ?? 'deg'),
      interval: 16,
    });
  }
  return out;
}
