/**
 * Synthetic phone-motion signals for the segmentation / controller tests (independent of src/sim,
 * which is written by another agent). Physics, all in the INITIAL device frame (= world frame W):
 *  - rotation about a fixed unit axis u by θ(t) = Σ angle_k·σ(s_k) + spin; R(θ) maps device → W.
 *  - translation p(t) = Σ dist_k·dir·σ(s_k)  ⇒  a_W = Σ dist_k·dir·σ''(s_k)/T_k²
 *  - σ(s) = 10s³ − 15s⁴ + 6s⁵ (minimum jerk), s = (t − start)/T clamped to [0, 1].
 *  - device readings: a_D = Rᵀa_W, up_D = Rᵀ·up_W, ω_D = u·θ' (a fixed axis is the same in both frames)
 *      acc  = gravitySign·a_D + noise
 *      accg = gravitySign·(a_D + g·up_D) + noise        (W3C: at rest accg points UP)
 *      rr   = {alpha: ω_x, beta: ω_y, gamma: ω_z} + tremor + noise   (schema 1.1 axis mapping)
 * Truth: a movement starts at `startMs` and ends (reaches the nose) at startMs + durMs.
 */
import type { RelMotionSample } from '../../src/core/segmentation';
import type { Euler, MotionSample, Vec3 } from '../../src/core/types';

export const G = 9.80665;

/** mulberry32 — a local copy so the tests depend on nothing but the module under test. */
export function segRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal deviate (Box–Muller). */
export function gauss(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Minimum-jerk position profile and its first two derivatives with respect to s. */
export function minJerk(s: number): { p: number; v: number; a: number } {
  if (s <= 0) return { p: 0, v: 0, a: 0 };
  if (s >= 1) return { p: 1, v: 0, a: 0 };
  const s2 = s * s;
  const s3 = s2 * s;
  return {
    p: 10 * s3 - 15 * s3 * s + 6 * s3 * s2,
    v: 30 * s2 - 60 * s3 + 30 * s3 * s,
    a: 60 * s - 180 * s2 + 120 * s3,
  };
}

function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

/** Rᵀ·v for R = rotation by `deg` about unit axis u (Rodrigues with −θ). */
export function rotateInv(u: Vec3, deg: number, v: Vec3): Vec3 {
  const th = (deg * Math.PI) / 180;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const cross: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  return [
    v[0] * c - cross[0] * s + u[0] * dot * (1 - c),
    v[1] * c - cross[1] * s + u[1] * dot * (1 - c),
    v[2] * c - cross[2] * s + u[2] * dot * (1 - c),
  ];
}

/** One minimum-jerk component: rotate by angleDeg about the model axis and translate distM. */
export interface SegMove {
  startMs: number;
  durMs: number;
  distM: number;
  angleDeg: number;
}

export interface SegModelOptions {
  seed: number;
  /** rotation axis, initial device frame (normalized internally) */
  axis?: Vec3;
  /** translation direction, initial device frame (normalized internally) */
  dir?: Vec3;
  /** world "up" in the initial device frame: [0,1,0] = upright, [1,0,0] = top_left (screen toward), [0,0,1] = flat */
  up?: Vec3;
  gravitySign?: 1 | -1;
  gyroNoiseDps?: number;
  accNoiseMs2?: number;
  /** amplitude (deg/s per axis) of an 8–12 Hz physiological tremor */
  tremorDps?: number;
  acc?: 'present' | 'null';
  accg?: 'present' | 'null';
  rr?: 'present' | 'null' | 'intermittent';
}

/** Signal generator; moves can be added while sampling (the fake source reacts to GO). */
export class SegMotionModel {
  readonly moves: SegMove[] = [];
  spin: { fromMs: number; dps: number } | null = null;
  private readonly rng: () => number;
  private readonly u: Vec3;
  private readonly dir: Vec3;
  private readonly up: Vec3;
  private readonly sign: 1 | -1;
  private readonly gyroNoise: number;
  private readonly accNoise: number;
  private readonly tremor: number;
  private readonly tremorHz: number;
  private readonly phase: Vec3;
  private readonly o: SegModelOptions;

  constructor(o: SegModelOptions) {
    this.o = o;
    this.rng = segRng(o.seed);
    this.u = normalize(o.axis ?? [1, 0.35, 0.2]);
    this.dir = normalize(o.dir ?? [0.1, 0.35, 1]);
    this.up = normalize(o.up ?? [0, 1, 0]);
    this.sign = o.gravitySign ?? 1;
    this.gyroNoise = o.gyroNoiseDps ?? 0.8;
    this.accNoise = o.accNoiseMs2 ?? 0.03;
    this.tremor = o.tremorDps ?? 1;
    this.tremorHz = 8 + 4 * this.rng();
    this.phase = [this.rng() * 6.283, this.rng() * 6.283, this.rng() * 6.283];
  }

  addMove(m: SegMove): void {
    this.moves.push(m);
  }

  /** Noise-free kinematics at time t (ms): rotation angle (deg), its rate (deg/s), world accel. */
  kinematics(t: number): { theta: number; thetaDot: number; aW: Vec3 } {
    let theta = 0;
    let thetaDot = 0;
    let aMag = 0;
    for (const m of this.moves) {
      const T = m.durMs / 1000;
      const mj = minJerk((t - m.startMs) / m.durMs);
      theta += m.angleDeg * mj.p;
      thetaDot += (m.angleDeg * mj.v) / T;
      aMag += (m.distM * mj.a) / (T * T);
    }
    if (this.spin && t > this.spin.fromMs) {
      theta += (this.spin.dps * (t - this.spin.fromMs)) / 1000;
      thetaDot += this.spin.dps;
    }
    return { theta, thetaDot, aW: [this.dir[0] * aMag, this.dir[1] * aMag, this.dir[2] * aMag] };
  }

  /** A full sample at source time t (noise drawn in a fixed order: deterministic per seed). */
  sampleAt(t: number): Omit<MotionSample, 't' | 'ts' | 'interval'> {
    const k = this.kinematics(t);
    const aD = rotateInv(this.u, k.theta, k.aW);
    const upD = rotateInv(this.u, k.theta, this.up);
    const sg = this.sign;
    const n = (): number => gauss(this.rng);
    const acc: Vec3 = [
      sg * aD[0] + this.accNoise * n(),
      sg * aD[1] + this.accNoise * n(),
      sg * aD[2] + this.accNoise * n(),
    ];
    const accg: Vec3 = [
      sg * (aD[0] + G * upD[0]) + this.accNoise * n(),
      sg * (aD[1] + G * upD[1]) + this.accNoise * n(),
      sg * (aD[2] + G * upD[2]) + this.accNoise * n(),
    ];
    const w = 2 * Math.PI * this.tremorHz * (t / 1000);
    const wx = this.u[0] * k.thetaDot + this.tremor * Math.sin(w + this.phase[0]) + this.gyroNoise * n();
    const wy = this.u[1] * k.thetaDot + this.tremor * Math.sin(w + this.phase[1]) + this.gyroNoise * n();
    const wz = this.u[2] * k.thetaDot + this.tremor * Math.sin(w + this.phase[2]) + this.gyroNoise * n();
    const rrMode = this.o.rr ?? 'present';
    const dropRr = rrMode === 'null' || (rrMode === 'intermittent' && this.rng() < 0.3);
    const rr: Euler | null = dropRr ? null : { alpha: wx, beta: wy, gamma: wz };
    return {
      acc: this.o.acc === 'null' ? null : acc,
      accg: this.o.accg === 'null' ? null : accg,
      rr,
    };
  }
}

export interface SegScenario extends SegModelOptions {
  hz?: number;
  startT?: number;
  durationMs: number;
  /** uniform ±jitter (ms) on each nominal sample time; must stay below half a period */
  jitterMs?: number;
  dropProb?: number;
  /** no samples with t in [a, b) */
  gaps?: Array<[number, number]>;
  moves?: SegMove[];
  spin?: { fromMs: number; dps: number } | null;
}

/** Relative samples for the segmenter (t already relative; index = position in the list). */
export function synthSamples(sc: SegScenario): RelMotionSample[] {
  const model = new SegMotionModel(sc);
  for (const m of sc.moves ?? []) model.addMove(m);
  model.spin = sc.spin ?? null;
  const timing = segRng((sc.seed ^ 0x5bd1e995) >>> 0);
  const hz = sc.hz ?? 60;
  const period = 1000 / hz;
  const start = sc.startT ?? 0;
  const out: RelMotionSample[] = [];
  for (let k = 0; ; k++) {
    const nominal = start + k * period;
    if (nominal > start + sc.durationMs) break;
    const t = nominal + (sc.jitterMs ? (2 * timing() - 1) * sc.jitterMs : 0);
    const drop = sc.dropProb ? timing() < sc.dropProb : false;
    const s = model.sampleAt(t);
    if (drop || (sc.gaps ?? []).some(([a, b]) => t >= a && t < b)) continue;
    out.push({ t, ts: t, interval: period, ...s, index: out.length });
  }
  return out;
}
