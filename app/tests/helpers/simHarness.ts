/**
 * Test helpers for the simulator: a minimal fake listener (stands in for the TrialController, which
 * may not be implemented yet) and a re-implementation of the smoothing part of the "online_v1"
 * segmentation spec, so stillness/onset checks do not depend on core/segmentation.ts.
 */
import type {
  MotionSample,
  OrientationSample,
  QuaternionSample,
  TrialCondition,
} from '../../src/core/types';
import {
  createSimulatedSource,
  type SimBehavior,
  type SimSourceOptions,
  type SimulatedSource,
} from '../../src/sim/simulatedSource';
import {
  defaultSimPlatform,
  randomSimParticipant,
  simRng,
  type SimParticipant,
  type SimPlatform,
} from '../../src/sim/model';

export const START_T = 1000;

export interface Collected {
  source: SimulatedSource;
  motion: MotionSample[];
  orientation: OrientationSample[];
  quaternion: QuaternionSample[];
}

export interface CollectOptions {
  /** give GO at the first motion sample with t − START_T ≥ goAtMs (null: never) */
  goAtMs?: number | null;
  /** stop this long after GO (or after start when no GO is given) */
  stopAfterMs?: number;
  /** send the 'end' cue this long after the true arrival at the nose (null: never) */
  endCueAfterArriveMs?: number | null;
}

/** Runs an OFFLINE source to completion with a scripted fake listener. */
export function collect(source: SimulatedSource, o: CollectOptions = {}): Collected {
  const goAt = o.goAtMs === undefined ? 1500 : o.goAtMs;
  const stopAfter = o.stopAfterMs ?? 6000;
  const endCueAfter = o.endCueAfterArriveMs ?? null;
  const out: Collected = { source, motion: [], orientation: [], quaternion: [] };
  let endSent = false;
  void source.start({
    onMotion(s) {
      out.motion.push(s);
      const rel = s.t - START_T;
      if (goAt !== null && source.truth.go_t === null && rel >= goAt) source.notifyCue?.('go', s.t);
      const arrive = source.truth.end_t;
      if (endCueAfter !== null && !endSent && arrive !== null && s.t >= arrive + endCueAfter) {
        endSent = true;
        source.notifyCue?.('end', s.t);
      }
      // with a planned GO, the stop clock starts at GO; otherwise at the start
      const ref = goAt === null ? START_T : source.truth.go_t;
      if (ref !== null && s.t - ref >= stopAfter) source.stop();
    },
    onOrientation(s) {
      out.orientation.push(s);
    },
    onQuaternion(s) {
      out.quaternion.push(s);
    },
  });
  return out;
}

export interface SimSetup {
  participant: SimParticipant;
  platform: SimPlatform;
}

/** A random participant + platform from a seed (deterministic). */
export function setup(seed: number, platform: 'ios' | 'android' = 'android'): SimSetup {
  const rng = simRng(seed);
  return { participant: randomSimParticipant(rng), platform: defaultSimPlatform(platform, rng) };
}

/** Same participant with dominance effects removed (needed for exact mirror comparisons). */
export function symmetric(p: SimParticipant): SimParticipant {
  return { ...p, style: { ...p.style, nondominant_slowdown: 1, nondominant_jitter: 1 } };
}

export function condition(over: Partial<TrialCondition> = {}): TrialCondition {
  return {
    hand: 'right',
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
    ...over,
  };
}

export function offlineSource(
  s: SimSetup,
  c: TrialCondition,
  seed: number,
  extra: Partial<SimSourceOptions> & { behavior?: SimBehavior } = {},
): SimulatedSource {
  return createSimulatedSource({
    mode: 'offline',
    participant: s.participant,
    platform: s.platform,
    condition: c,
    seed,
    startTime: START_T,
    baseEpochMs: 1_700_000_000_000,
    ...extra,
  });
}

export function norm3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

export function gyroMag(s: MotionSample): number {
  return s.rr ? norm3(s.rr.alpha, s.rr.beta, s.rr.gamma) : 0;
}

export function accMag(s: MotionSample): number {
  return s.acc ? norm3(s.acc[0], s.acc[1], s.acc[2]) : 0;
}

/**
 * Smoothed |ω| and |a_lin| exactly as in the online_v1 spec: EMA with k_i = 1 − exp(−dt_i/τ),
 * S_0 = x_0; acc source 'linear' when the first sample has acc, else 'highpass' (EMA gravity with
 * gravityTauMs, gravity updated BEFORE subtracting).
 */
export function smoothed(
  samples: MotionSample[],
  smoothingMs = 40,
  gravityTauMs = 600,
): { t: number[]; gyro: number[]; acc: number[] } {
  const t: number[] = [];
  const gyro: number[] = [];
  const acc: number[] = [];
  const linear = samples.length > 0 && samples[0].acc !== null;
  let g = 0;
  let a = 0;
  let G: [number, number, number] | null = null;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    const dt = i === 0 ? 0 : s.t - samples[i - 1].t;
    const rawG = gyroMag(s);
    let rawA: number;
    if (linear) rawA = accMag(s);
    else {
      const v = s.accg as [number, number, number];
      if (G === null) G = [v[0], v[1], v[2]];
      else {
        const k = 1 - Math.exp(-dt / gravityTauMs);
        G = [G[0] + k * (v[0] - G[0]), G[1] + k * (v[1] - G[1]), G[2] + k * (v[2] - G[2])];
      }
      rawA = norm3(v[0] - G[0], v[1] - G[1], v[2] - G[2]);
    }
    if (i === 0) {
      g = rawG;
      a = rawA;
    } else {
      const k = smoothingMs === 0 ? 1 : 1 - Math.exp(-dt / smoothingMs);
      g += k * (rawG - g);
      a += k * (rawA - a);
    }
    t.push(s.t);
    gyro.push(g);
    acc.push(a);
  }
  return { t, gyro, acc };
}
