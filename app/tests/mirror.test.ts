import { describe, expect, it } from 'vitest';
import { ALL_ORIENTATIONS, GRAVITY } from '../src/core/constants';
import {
  eulerZXYFromMat,
  mExp,
  mLog,
  mMul,
  mT,
  mVec,
  matFromEulerZXY,
  matFromQuat,
  rotationRateToVector,
  vectorToRotationRate,
  type Mat3,
} from '../src/core/math3d';
import { mirrorCondition, mirrorMotionSample, mirrorOrientationSample, mirrorTrial } from '../src/core/mirror';
import type { MotionSample, OrientationSample, TrialCondition, Vec3 } from '../src/core/types';
import { simRng } from '../src/sim/model';
import { expectEqualColumnLengths, expectValid } from './helpers/schema';
import { trialFromCollected } from './helpers/simFixtures';
import { collect, condition, offlineSource, setup, symmetric } from './helpers/simHarness';

const M: Mat3 = [-1, 0, 0, 0, 1, 0, 0, 0, 1];
const rng = simRng(777);

function randRotation(r: () => number): Mat3 {
  const g = (): number => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const q: [number, number, number, number] = [g(), g(), g(), g()];
  return matFromQuat(q);
}

function expectMatClose(a: Mat3, b: Mat3, tol = 1e-9): void {
  for (let i = 0; i < 9; i++) expect(Math.abs(a[i] - b[i]), `entry ${i}`).toBeLessThan(tol);
}

function expectVecClose(a: Vec3, b: Vec3, tol: number): void {
  for (let i = 0; i < 3; i++) expect(Math.abs(a[i] - b[i]), `component ${i}: ${a} vs ${b}`).toBeLessThan(tol);
}

function angDiff(a: number, b: number): number {
  return ((((a - b) % 360) + 540) % 360) - 180;
}

describe('mirrorCondition', () => {
  it('swaps the hand, maps the orientation through ORIENTATIONS.mirror and keeps the rest', () => {
    const c: TrialCondition = condition({ hand: 'right', orientation: 'top_left_toward', speed: 'fast', variant: 'twist', rep: 2 });
    expect(mirrorCondition(c)).toEqual({ ...c, hand: 'left', orientation: 'top_right_toward' });
    expect(mirrorCondition(condition({ orientation: 'upright_away' })).orientation).toBe('upright_away');
    expect(mirrorCondition(condition({ orientation: 'top_right_away', hand: 'left' }))).toMatchObject({ hand: 'right', orientation: 'top_left_away' });
  });

  it('is an involution for every orientation and hand', () => {
    for (const o of ALL_ORIENTATIONS) {
      for (const hand of ['left', 'right'] as const) {
        const c = condition({ orientation: o, hand });
        expect(mirrorCondition(mirrorCondition(c))).toEqual(c);
      }
    }
  });
});

describe('mirrorMotionSample', () => {
  const s: MotionSample = {
    t: 12.5,
    ts: 10,
    acc: [1, 2, 3],
    accg: [4, 5, 6],
    rr: { alpha: 7, beta: 8, gamma: 9 },
    interval: 16,
  };

  it('a → (−ax, ay, az); rotationRate alpha (x) kept, beta (y) and gamma (z) negated', () => {
    expect(mirrorMotionSample(s)).toEqual({
      t: 12.5,
      ts: 10,
      acc: [-1, 2, 3],
      accg: [-4, 5, 6],
      rr: { alpha: 7, beta: -8, gamma: -9 },
      interval: 16,
    });
  });

  it('keeps nulls and is an exact involution', () => {
    const n: MotionSample = { t: 1, ts: null, acc: null, accg: [0.5, 9.8, -0.1], rr: null, interval: null };
    expect(mirrorMotionSample(n)).toEqual({ ...n, accg: [-0.5, 9.8, -0.1] });
    expect(mirrorMotionSample(mirrorMotionSample(s))).toEqual(s);
    expect(mirrorMotionSample(mirrorMotionSample(n))).toEqual(n);
    // the input is not modified
    expect(s.acc).toEqual([1, 2, 3]);
  });
});

describe('mirrorOrientationSample', () => {
  const base: OrientationSample = { t: 5, ts: 4, alpha: 0, beta: 0, gamma: 0, absolute: false, compass: null };

  it('equals R′ = Mw·R·S, i.e. (360 − alpha, beta, −gamma) for generic angles', () => {
    for (let i = 0; i < 500; i++) {
      const alpha = rng() * 360;
      const beta = rng() * 340 - 170;
      const gamma = rng() * 178 - 89;
      if (Math.abs(Math.abs(beta) - 90) < 1) continue;
      const m = mirrorOrientationSample({ ...base, alpha, beta, gamma });
      const R = matFromEulerZXY(alpha, beta, gamma);
      expectMatClose(matFromEulerZXY(m.alpha!, m.beta!, m.gamma!), mMul(mMul(M, R), M), 1e-9);
      expect(Math.abs(angDiff(m.alpha!, 360 - alpha))).toBeLessThan(1e-8);
      expect(Math.abs(m.beta! - beta)).toBeLessThan(1e-8);
      expect(Math.abs(m.gamma! + gamma)).toBeLessThan(1e-8);
    }
  });

  it('handles gamma = −90 (whose naive mirror +90 is outside the W3C range)', () => {
    const m = mirrorOrientationSample({ ...base, alpha: 30, beta: 20, gamma: -90 });
    expect(m.gamma!).toBeGreaterThanOrEqual(-90);
    expect(m.gamma!).toBeLessThan(90);
    expectMatClose(matFromEulerZXY(m.alpha!, m.beta!, m.gamma!), mMul(mMul(M, matFromEulerZXY(30, 20, -90)), M), 1e-9);
  });

  it('is an involution (as a rotation, also at gimbal lock)', () => {
    for (let i = 0; i < 500; i++) {
      const e = eulerZXYFromMat(randRotation(rng));
      const s: OrientationSample = { ...base, ...e, compass: rng() * 360 };
      const back = mirrorOrientationSample(mirrorOrientationSample(s));
      expectMatClose(matFromEulerZXY(back.alpha!, back.beta!, back.gamma!), matFromEulerZXY(e.alpha, e.beta, e.gamma), 1e-9);
      expect(Math.abs(angDiff(back.compass!, s.compass!))).toBeLessThan(1e-9);
    }
    const g = mirrorOrientationSample(mirrorOrientationSample({ ...base, alpha: 40, beta: 90, gamma: 10 }));
    expectMatClose(matFromEulerZXY(g.alpha!, g.beta!, g.gamma!), matFromEulerZXY(40, 90, 10), 1e-9);
  });

  it('nulls stay null (all three angles when one is missing); compass → 360 − heading', () => {
    expect(mirrorOrientationSample({ ...base, alpha: null, beta: 10, gamma: 5 })).toMatchObject({ alpha: null, beta: null, gamma: null });
    expect(mirrorOrientationSample({ ...base, alpha: null, beta: null, gamma: null, compass: 90 })).toEqual({
      ...base,
      alpha: null,
      beta: null,
      gamma: null,
      compass: 270,
    });
    expect(mirrorOrientationSample({ ...base, compass: 0 }).compass).toBe(0);
    expect(mirrorOrientationSample({ ...base, compass: 359 }).compass).toBeCloseTo(1, 12);
  });
});

describe('mirror derivation is physically consistent', () => {
  it('gravity: S·(Rᵀ·up) equals the accelerometer reading of the mirrored pose Mw·R·S', () => {
    for (let i = 0; i < 200; i++) {
      const R = randRotation(rng);
      const up: Vec3 = [0, 0, GRAVITY];
      const accg = mVec(mT(R), up);
      const e = eulerZXYFromMat(R);
      const m = mirrorOrientationSample({ t: 0, ts: 0, ...e, absolute: false, compass: null });
      const Rm = matFromEulerZXY(m.alpha!, m.beta!, m.gamma!);
      const s: MotionSample = { t: 0, ts: 0, acc: null, accg, rr: null, interval: null };
      expectVecClose(mirrorMotionSample(s).accg!, mVec(mT(Rm), up), 1e-9);
    }
  });

  it('angular velocity: the gyro of the mirrored trajectory is −S·ω (a pseudo-vector)', () => {
    const h = 1e-4;
    for (let i = 0; i < 100; i++) {
      const R0 = randRotation(rng);
      const w: Vec3 = [rng() * 4 - 2, rng() * 4 - 2, rng() * 4 - 2]; // device-frame rad/s
      const R1 = mMul(R0, mExp([w[0] * h, w[1] * h, w[2] * h]));
      // device-frame angular velocity from the orientation change: Rᵀ·Ṙ
      const wD = mLog(mMul(mT(R0), R1)).map((v) => v / h) as Vec3;
      const R0m = mMul(mMul(M, R0), M);
      const R1m = mMul(mMul(M, R1), M);
      const wDm = mLog(mMul(mT(R0m), R1m)).map((v) => v / h) as Vec3;
      const toDps = (v: Vec3): Vec3 => v.map((x) => (x * 180) / Math.PI) as Vec3;
      const sample: MotionSample = { t: 0, ts: 0, acc: null, accg: null, rr: vectorToRotationRate(toDps(wD)), interval: null };
      const mirrored = mirrorMotionSample(sample).rr!;
      expectVecClose(rotationRateToVector(mirrored), toDps(wDm), 1e-6);
      // spelled out in W3C 2025 field names: alpha = x kept, beta = y and gamma = z negated
      expect(mirrored.alpha).toBe(toDps(wD)[0]);
      expect(mirrored.beta).toBe(-toDps(wD)[1]);
      expect(mirrored.gamma).toBe(-toDps(wD)[2]);
    }
  });
});

describe('mirrorTrial', () => {
  const s = setup(4242);
  const run = (cond: TrialCondition, heading = 0) => {
    const st = { participant: symmetric(s.participant), platform: { ...s.platform, heading_deg: heading } };
    return collect(offlineSource(st, cond, 99, { noise: false }), { goAtMs: 1200, stopAfterMs: 2500 });
  };

  it('returns a mirrored, schema-valid deep copy and leaves the input untouched', () => {
    const cond = condition({ hand: 'right', orientation: 'top_left_toward' });
    const trial = trialFromCollected(run(cond), cond);
    expectValid('TrialRecord', trial);
    const before = structuredClone(trial);
    const m = mirrorTrial(trial);
    expect(trial).toEqual(before);
    expectValid('TrialRecord', m);
    expectEqualColumnLengths(m.streams.motion as unknown as Record<string, unknown>);
    expectEqualColumnLengths(m.streams.orientation as unknown as Record<string, unknown>);
    expectEqualColumnLengths(m.streams.quaternion as unknown as Record<string, unknown>);
    expect(m.condition).toEqual({ ...cond, hand: 'left', orientation: 'top_right_toward' });
    expect(m.detected.rotation_start).toBe('top_right');
    expect(m.detected.rotation_end).toBe('upright');
    expect(m.segmentation.baseline!.gravity).toEqual([-1.5, 9.6, 0.4]);
    expect(m.screen).toEqual({ type_start: 'landscape-secondary', angle_start: 270, type_end: 'portrait-primary', angle_end: 0 });
    expect(m.streams.touches).toEqual(trial.streams.touches);
    for (let i = 0; i < trial.streams.motion.t.length; i++) {
      expect(m.streams.motion.acc_x[i]).toBe(-trial.streams.motion.acc_x[i]!);
      expect(m.streams.motion.acc_y[i]).toBe(trial.streams.motion.acc_y[i]);
      expect(m.streams.motion.accg_x[i]).toBe(-trial.streams.motion.accg_x[i]!);
      expect(m.streams.motion.rr_alpha[i]).toBe(trial.streams.motion.rr_alpha[i]); // x
      expect(m.streams.motion.rr_beta[i]).toBe(-trial.streams.motion.rr_beta[i]!); // y
      expect(m.streams.motion.rr_gamma[i]).toBe(-trial.streams.motion.rr_gamma[i]!); // z
    }
  });

  it('schema 1.2 streams: absolute orientation / quaternion like their relative twins, magnetometer x negated, screen events reflected', () => {
    const cond = condition({ hand: 'right', orientation: 'top_left_toward' });
    const base = trialFromCollected(run(cond, 37), cond);
    const o = base.streams.orientation!;
    const q = base.streams.quaternion!;
    const trial = structuredClone(base);
    trial.streams.orientation_abs = { ...structuredClone(o), absolute: true, compass: o.t.map((_, i) => (i % 2 ? 90.5 : null)), compass_accuracy: o.t.map(() => 12) };
    trial.streams.quaternion_abs = { ...structuredClone(q), sensor: 'AbsoluteOrientationSensor' };
    trial.streams.magnetometer = { t: [5, 21.5], x: [20.5, null], y: [-30, -29.5], z: [15.25, 14], frequency_hz: 60 };
    trial.streams.screen_events = [
      { t: 100, type: 'landscape-primary', angle: 90 },
      { t: 200, type: 'portrait-primary', angle: 0 },
      { t: 300, type: null, angle: null },
    ];
    trial.streams.touches = [{ t: 1200, type: 'start', n: 1, x: 100, y: 200, radius_x: 10, radius_y: 8, force: 0.5, rotation_angle: 20, x_norm: 0.25, y_norm: 0.3 }];
    expectValid('TrialRecord', trial);
    const m = mirrorTrial(trial);
    expectValid('TrialRecord', m);
    const ma = m.streams.orientation_abs!;
    const mo = m.streams.orientation!;
    expect(ma.absolute).toBe(true);
    expect(ma.compass).toEqual(o.t.map((_, i) => (i % 2 ? 269.5 : null)));
    expect(ma.compass_accuracy).toEqual(o.t.map(() => 12)); // an uncertainty, not a direction
    for (let i = 0; i < o.t.length; i++) {
      // the same angles as the mirrored relative stream (they were copies before mirroring)
      expect([ma.alpha[i], ma.beta[i], ma.gamma[i]]).toEqual([mo.alpha[i], mo.beta[i], mo.gamma[i]]);
    }
    const mq = m.streams.quaternion_abs!;
    expect(mq.sensor).toBe('AbsoluteOrientationSensor');
    expect(mq.x).toEqual(q.x);
    expect(mq.y).toEqual(q.y.map((v) => -v));
    expect(mq.z).toEqual(q.z.map((v) => -v));
    expect(mq.w).toEqual(q.w);
    expect(m.streams.magnetometer).toEqual({ t: [5, 21.5], x: [-20.5, null], y: [-30, -29.5], z: [15.25, 14], frequency_hz: 60 });
    expect(m.streams.screen_events).toEqual([
      { t: 100, type: 'landscape-secondary', angle: 270 },
      { t: 200, type: 'portrait-primary', angle: 0 },
      { t: 300, type: null, angle: null },
    ]);
    expect(m.streams.touches).toEqual(trial.streams.touches); // touches are not mirrored
    // involution: exact for everything but the Euler angles (compared as rotations above)
    const back = mirrorTrial(m);
    expect(back.streams.quaternion_abs).toEqual(trial.streams.quaternion_abs);
    expect(back.streams.magnetometer).toEqual(trial.streams.magnetometer);
    expect(back.streams.screen_events).toEqual(trial.streams.screen_events);
    expect(back.streams.orientation_abs?.compass).toEqual(trial.streams.orientation_abs.compass);
    // older records without the schema 1.2 streams stay without them
    const old = mirrorTrial(base);
    expect('orientation_abs' in old.streams || 'magnetometer' in old.streams || 'screen_events' in old.streams).toBe(false);
  });

  it('is an involution', () => {
    const cond = condition({ hand: 'left', orientation: 'upright_away', speed: 'fast' });
    const trial = trialFromCollected(run(cond, 123), cond);
    const back = mirrorTrial(mirrorTrial(trial));
    expect(back.condition).toEqual(trial.condition);
    expect(back.streams.motion).toEqual(trial.streams.motion);
    expect(back.streams.quaternion).toEqual(trial.streams.quaternion);
    expect(back.screen).toEqual(trial.screen);
    const o = trial.streams.orientation!;
    const ob = back.streams.orientation!;
    for (let i = 0; i < o.t.length; i++) {
      expectMatClose(matFromEulerZXY(ob.alpha[i]!, ob.beta[i]!, ob.gamma[i]!), matFromEulerZXY(o.alpha[i]!, o.beta[i]!, o.gamma[i]!), 1e-9);
    }
  });

  it('mirroring a simulated right-hand trial reproduces the simulated left-hand trial (heading 0)', () => {
    for (const o of ['top_left_toward', 'upright_toward', 'top_right_away', 'flat_up', 'free'] as const) {
      const right = condition({ hand: 'right', orientation: o, variant: 'curved' });
      const left = mirrorCondition(right);
      const mr = mirrorTrial(trialFromCollected(run(right), right));
      const l = trialFromCollected(run(left), left);
      expect(mr.condition).toEqual(l.condition);
      const a = mr.streams.motion;
      const b = l.streams.motion;
      expect(a.t).toEqual(b.t);
      // non-vacuous: a full movement with large lateral components is being compared
      expect(a.t.length).toBeGreaterThan(150);
      expect(Math.max(...b.rr_beta.map((v) => Math.abs(v!)), ...b.rr_gamma.map((v) => Math.abs(v!)))).toBeGreaterThan(20);
      expect(Math.max(...b.acc_x.map((v) => Math.abs(v!)), ...b.acc_y.map((v) => Math.abs(v!)))).toBeGreaterThan(0.5);
      for (const k of ['acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma'] as const) {
        for (let i = 0; i < a.t.length; i++) expect(Math.abs(a[k][i]! - b[k][i]!), `${o} ${k}[${i}]`).toBeLessThan(1e-9);
      }
      const ao = mr.streams.orientation!;
      const bo = l.streams.orientation!;
      for (let i = 0; i < ao.t.length; i++) {
        expectMatClose(matFromEulerZXY(ao.alpha[i]!, ao.beta[i]!, ao.gamma[i]!), matFromEulerZXY(bo.alpha[i]!, bo.beta[i]!, bo.gamma[i]!), 1e-9);
      }
      const aq = mr.streams.quaternion!;
      const bq = l.streams.quaternion!;
      for (let i = 0; i < aq.t.length; i++) {
        const qa = matFromQuat([aq.x[i], aq.y[i], aq.z[i], aq.w[i]]);
        const qb = matFromQuat([bq.x[i], bq.y[i], bq.z[i], bq.w[i]]);
        expectMatClose(qa, qb, 1e-9);
      }
    }
  });
});
