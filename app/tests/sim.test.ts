import { describe, expect, it } from 'vitest';
import { ALL_ORIENTATIONS, GRAVITY, ORIENTATIONS } from '../src/core/constants';
import {
  DEG,
  angleBetweenDeg,
  mDet,
  mExp,
  mIdentity,
  mLog,
  mMul,
  mT,
  mVec,
  matFromEulerZXY,
  matFromQuat,
  rotationRateToVector,
  vNorm,
  type Mat3,
} from '../src/core/math3d';
import { mirrorCondition, mirrorMotionSample } from '../src/core/mirror';
import type {
  MotionSample,
  OrientationId,
  OrientationSample,
  ParticipantProfile,
  QuaternionSample,
  Speed,
  TrialCondition,
  Variant,
  Vec3,
} from '../src/core/types';
import {
  GRIP_ORIENTATIONS,
  IOS_GRAVITY_SIGN,
  defaultSimPlatform,
  earthFromBody,
  gripMatrix,
  performedCondition,
  planMovement,
  randomSimParticipant,
  simChoiceHand,
  simParticipantFromProfile,
  simRng,
  type GripOrientation,
} from '../src/sim/model';
import { FALSE_START_AFTER_STILL_MS, createSimulatedSource } from '../src/sim/simulatedSource';
import {
  START_T,
  accMag,
  collect,
  condition,
  gyroMag,
  offlineSource,
  setup,
  smoothed,
  symmetric,
  type Collected,
} from './helpers/simHarness';

const S: Mat3 = [-1, 0, 0, 0, 1, 0, 0, 0, 1];
const D = DEFAULTS();

function DEFAULTS() {
  // DEFAULT_SEGMENTATION_PARAMS values the simulator is designed around (kept local on purpose:
  // the test documents the contract even if the defaults are tuned later)
  return { stillGyro: 15, stillAcc: 0.35, onsetGyro: 30, onsetAcc: 0.6, endGyro: 20, endAcc: 0.45 };
}

function expectMatClose(a: Mat3, b: Mat3, tol = 1e-12): void {
  for (let i = 0; i < 9; i++) expect(Math.abs(a[i] - b[i]), `entry ${i}: ${a} vs ${b}`).toBeLessThan(tol);
}

function maxOver(samples: MotionSample[], f: (s: MotionSample) => number, from = -Infinity, to = Infinity): number {
  let m = 0;
  for (const s of samples) if (s.ts! >= from && s.ts! <= to) m = Math.max(m, f(s));
  return m;
}

function run(seed: number, c: TrialCondition, opts: { noise?: boolean; platform?: 'ios' | 'android'; sym?: boolean; heading?: number } = {}, collectOpts = {}): Collected {
  const st = setup(seed, opts.platform ?? 'android');
  const s2 = {
    participant: opts.sym ? symmetric(st.participant) : st.participant,
    platform: opts.heading === undefined ? st.platform : { ...st.platform, heading_deg: opts.heading },
  };
  return collect(offlineSource(s2, c, seed + 1, { noise: opts.noise ?? false }), collectOpts);
}

// ───────────────────────────── frames and grips ─────────────────────────────

describe('grips and frames', () => {
  it('every grip is a proper rotation (det +1) and M·grip(o)·S = grip(mirror(o))', () => {
    for (const o of GRIP_ORIENTATIONS) {
      const g = gripMatrix(o);
      expect(mDet(g)).toBe(1);
      expectMatClose(mMul(g, mT(g)), mIdentity(), 0.5);
      expectMatClose(mMul(g, mT(g)), mIdentity(), 1e-15);
      const mirrored = ORIENTATIONS[o].mirror as GripOrientation;
      expect(mMul(mMul(S, g), S)).toEqual(gripMatrix(mirrored).map((v) => v)); // M = S numerically
    }
  });

  it('screen facing: device z points backward (to the face) for toward grips and forward for away grips', () => {
    for (const o of GRIP_ORIENTATIONS) {
      const z = mVec(gripMatrix(o), [0, 0, 1]);
      const facing = ORIENTATIONS[o].screen;
      if (facing === 'toward') expect(z).toEqual([0, 0, 1]);
      if (facing === 'away') expect(z).toEqual([0, 0, -1]);
      if (facing === 'up') expect(z).toEqual([0, 1, 0]);
    }
  });

  it('earthFromBody: right-handed, facing North → right = East; facing East → forward = East', () => {
    const r0 = earthFromBody(0);
    expect(mDet(r0)).toBeCloseTo(1, 15);
    expect(mVec(r0, [1, 0, 0]).map((v) => +v.toFixed(12))).toEqual([1, 0, 0]);
    expect(mVec(r0, [0, 0, -1]).map((v) => +v.toFixed(12))).toEqual([0, 1, 0]); // forward = −z body = North
    expect(mVec(r0, [0, 1, 0])).toEqual([0, 0, 1]);
    const r90 = earthFromBody(90);
    expect(mVec(r90, [0, 0, -1]).map((v) => +v.toFixed(12))).toEqual([1, 0, 0]);
    expect(mVec(r90, [1, 0, 0]).map((v) => +v.toFixed(12))).toEqual([0, -1, 0]); // right = South
  });

  it('at rest, accg (gravitySign +1) points UP in the device frame', () => {
    const expected: Record<GripOrientation, Vec3> = {
      upright_toward: [0, GRAVITY, 0],
      upright_away: [0, GRAVITY, 0],
      upside_down_toward: [0, -GRAVITY, 0],
      upside_down_away: [0, -GRAVITY, 0],
      top_left_toward: [GRAVITY, 0, 0],
      top_left_away: [-GRAVITY, 0, 0],
      top_right_toward: [-GRAVITY, 0, 0],
      top_right_away: [GRAVITY, 0, 0],
      flat_up: [0, 0, GRAVITY],
    };
    for (const o of GRIP_ORIENTATIONS) {
      // exact for the nominal grip ...
      expect(mVec(mT(gripMatrix(o)), [0, GRAVITY, 0])).toEqual(expected[o]);
      // ... and within the ≤ 10° per-trial grip tilt for the simulated source, both hands
      for (const hand of ['right', 'left'] as const) {
        const c = run(31, condition({ orientation: o, hand }), {}, { goAtMs: null, stopAfterMs: 400 });
        expect(c.motion.length).toBeGreaterThan(10);
        for (const s of c.motion) {
          expect(angleBetweenDeg(s.accg!, expected[o])).toBeLessThan(10.0001);
          expect(vNorm(s.accg!)).toBeCloseTo(GRAVITY, 9);
          expect(vNorm(s.acc!)).toBeLessThan(1e-12);
          expect(gyroMag(s)).toBeLessThan(1e-9);
        }
      }
    }
  });
});

// ───────────────────────────── THE KEY PROPERTY ─────────────────────────────

describe('mirror symmetry of the simulator (validates the mirror derivation)', () => {
  const speeds: Speed[] = ['natural', 'fast', 'slow'];
  const variants: Variant[] = ['standard', 'curved', 'pause', 'twist', 'elbow_high', 'jerky', 'lean_in', 'chin'];

  function pair(o: OrientationId, speed: Speed, variant: Variant, start: TrialCondition['start_position'], seed: number) {
    const right = condition({ hand: 'right', orientation: o, speed, variant, start_position: start });
    const left = mirrorCondition(right);
    const opts = { sym: true, noise: false, heading: 0 };
    const co = { goAtMs: 1300, endCueAfterArriveMs: 350, stopAfterMs: 9000 };
    return { r: run(seed, right, opts, co), l: run(seed, left, opts, co), left };
  }

  it('simulating (left, mirror(o)) equals mirror(simulating (right, o)) sample by sample', () => {
    let k = 0;
    for (const o of ALL_ORIENTATIONS) {
      for (let j = 0; j < 3; j++) {
        const speed = speeds[(k + j) % speeds.length];
        const variant = variants[(3 * k + j) % variants.length];
        const start = (['natural', 'shoulder', 'center'] as const)[(k + 2 * j) % 3];
        const seed = 500 + 17 * k + j;
        const { r, l, left } = pair(o, speed, variant, start, seed);
        expect(l.source.plan.orientation, `${o}`).toBe(o === 'free' ? ORIENTATIONS[r.source.plan.orientation].mirror : left.orientation);
        expect(l.motion.length).toBe(r.motion.length);
        expect(l.source.truth).toEqual(r.source.truth);
        // non-vacuous: the movement and the return happened inside the compared window
        expect(maxOver(r.motion, gyroMag), `${o} ${speed} ${variant}`).toBeGreaterThan(15);
        expect(r.source.truth.return_t).not.toBeNull();
        for (let i = 0; i < r.motion.length; i++) {
          const m = mirrorMotionSample(r.motion[i]);
          const s = l.motion[i];
          expect(s.t).toBe(m.t);
          for (let a = 0; a < 3; a++) {
            expect(Math.abs(s.acc![a] - m.acc![a]), `${o}/${speed}/${variant} acc[${a}] #${i}`).toBeLessThan(1e-9);
            expect(Math.abs(s.accg![a] - m.accg![a]), `${o}/${speed}/${variant} accg[${a}] #${i}`).toBeLessThan(1e-9);
          }
          expect(Math.abs(s.rr!.alpha - m.rr!.alpha), `${o} alpha #${i}`).toBeLessThan(1e-9);
          expect(Math.abs(s.rr!.beta - m.rr!.beta), `${o} beta #${i}`).toBeLessThan(1e-9);
          expect(Math.abs(s.rr!.gamma - m.rr!.gamma), `${o} gamma #${i}`).toBeLessThan(1e-9);
        }
      }
      k++;
    }
  });

  it('also holds for the false-start twitch and the too_slow wobble', () => {
    for (const behavior of ['false_start', 'too_slow'] as const) {
      const right = condition({ hand: 'right', orientation: 'top_left_away' });
      const st = setup(77);
      const sym = { participant: symmetric(st.participant), platform: { ...st.platform, heading_deg: 0 } };
      const co = { goAtMs: 1500, stopAfterMs: 3000 };
      const r = collect(offlineSource(sym, right, 5, { noise: false, behavior }), co);
      const l = collect(offlineSource(sym, mirrorCondition(right), 5, { noise: false, behavior }), co);
      expect(l.motion.length).toBe(r.motion.length);
      for (let i = 0; i < r.motion.length; i++) {
        const m = mirrorMotionSample(r.motion[i]);
        const s = l.motion[i];
        for (let a = 0; a < 3; a++) expect(Math.abs(s.accg![a] - m.accg![a])).toBeLessThan(1e-9);
        const d = rotationRateToVector(s.rr!).map((v, a) => Math.abs(v - rotationRateToVector(m.rr!)[a]));
        expect(Math.max(...d)).toBeLessThan(1e-9);
      }
    }
  });

  it('with a non-dominant hand the pair is NOT symmetric (dominance is a real confound)', () => {
    const st = setup(9);
    const p = { ...st.participant, dominant: 'right' as const };
    let slower = 0;
    for (let i = 0; i < 20; i++) {
      const c = condition({ hand: 'right' });
      const dr = planMovement(p, c, simRng(i)).duration_s;
      const dl = planMovement(p, mirrorCondition(c), simRng(i)).duration_s;
      if (dl > dr) slower++;
    }
    expect(slower).toBe(20); // same draws, the left (non-dominant) hand is always slower
  });
});

// ───────────────────────────── plausibility ─────────────────────────────

describe('kinematic plausibility', () => {
  it('natural speed peaks at roughly 60–400 deg/s and 1–12 m/s²; fast > natural > slow; durations match the plan', () => {
    let inside = 0;
    for (let seed = 1; seed <= 25; seed++) {
      const st = setup(1000 + seed);
      const hand = st.participant.dominant;
      const res: Record<Speed, { w: number; a: number; dur: number }> = {} as never;
      for (const speed of ['natural', 'fast', 'slow'] as const) {
        const c = collect(offlineSource(st, condition({ hand, speed }), 3 * seed, { noise: false }), { goAtMs: 1000, stopAfterMs: 6000 });
        const tr = c.source.truth;
        const dur = tr.end_t! - tr.onset_t!;
        expect(dur).toBeCloseTo(c.source.plan.duration_s * 1000, 9);
        // the phone does not move before the onset nor after the contact transient
        expect(maxOver(c.motion, gyroMag, -Infinity, tr.onset_t!)).toBeLessThan(1e-9);
        expect(maxOver(c.motion, accMag, -Infinity, tr.onset_t!)).toBeLessThan(1e-12);
        expect(maxOver(c.motion, gyroMag, tr.end_t! + 1, Infinity)).toBeLessThan(1e-6);
        expect(maxOver(c.motion, accMag, tr.end_t! + 61, Infinity)).toBeLessThan(1e-9);
        res[speed] = { w: maxOver(c.motion, gyroMag), a: maxOver(c.motion, accMag, tr.onset_t!, tr.end_t!), dur };
      }
      // "roughly": every participant within [45, 400], at least 90 % within [60, 400]
      expect(res.natural.w, `seed ${seed}`).toBeGreaterThan(45);
      expect(res.natural.w, `seed ${seed}`).toBeLessThan(400);
      if (res.natural.w >= 60) inside++;
      expect(res.natural.a, `seed ${seed}`).toBeGreaterThan(1);
      expect(res.natural.a, `seed ${seed}`).toBeLessThan(12);
      expect(res.fast.w).toBeGreaterThan(res.natural.w);
      expect(res.natural.w).toBeGreaterThan(res.slow.w);
      expect(res.fast.a).toBeGreaterThan(res.natural.a);
      expect(res.natural.a).toBeGreaterThan(res.slow.a);
      expect(res.fast.dur).toBeLessThan(res.natural.dur);
      expect(res.natural.dur).toBeLessThan(res.slow.dur);
      expect(res.natural.dur).toBeGreaterThan(500);
      expect(res.natural.dur).toBeLessThan(2000);
      expect(res.fast.dur).toBeGreaterThan(300);
    }
    expect(inside).toBeGreaterThanOrEqual(23);
  });

  it('the phone travels from p0 to the nose: start/end positions and orientation change', () => {
    const st = setup(55);
    for (const hand of ['right', 'left'] as const) {
      for (const start of ['shoulder', 'center', 'natural'] as const) {
        const plan = planMovement(st.participant, condition({ hand, start_position: start }), simRng(3));
        const h = hand === 'right' ? 1 : -1;
        const a = plan.pose(0);
        const b = plan.pose(plan.duration_s);
        expect(a.p).toEqual(plan.p0);
        for (let i = 0; i < 3; i++) expect(b.p[i]).toBeCloseTo(plan.p1[i], 12);
        expect(vNorm(a.v) + vNorm(b.v) + vNorm(a.a) + vNorm(b.a)).toBeLessThan(1e-12);
        expect(plan.p0[2]).toBeLessThan(-0.35); // arm extended in front
        expect(Math.abs(plan.p1[2])).toBeLessThan(0.03); // at the nose
        if (start === 'shoulder') expect(h * plan.p0[0]).toBeGreaterThan(0.12);
        if (start === 'center') expect(Math.abs(plan.p0[0])).toBeCloseTo(0.03, 10);
        expect(plan.pose(-1).p).toEqual(plan.p0); // clamped outside [0, T]
        // the net rotation over the movement is substantial but not absurd
        const turn = vNorm(mVecAngle(mMul(b.R, mT(a.R))));
        expect(turn).toBeGreaterThan(10);
        expect(turn).toBeLessThan(120);
      }
    }
  });

  it('variants change the plan as documented', () => {
    const p = setup(66).participant;
    const plan = (v: Variant, extra: Partial<TrialCondition> = {}) => planMovement(p, condition({ hand: p.dominant, variant: v, ...extra }), simRng(42));
    const std = plan('standard');
    expect(plan('chin').p1[1]).toBeCloseTo(std.p1[1] - 0.07, 12);
    expect(plan('forehead').p1[1]).toBeCloseTo(std.p1[1] + 0.06, 12);
    expect(plan('lean_in').p1[2]).toBeCloseTo(std.p1[2] - 0.08, 12);
    expect(plan('lean_in').duration_s).toBeCloseTo(std.duration_s * 0.9, 12);
    expect(plan('bent_start').p0[2]).toBeGreaterThan(std.p0[2] + 0.08); // L × 0.8
    expect(plan('pause').duration_s).toBeCloseTo(std.duration_s + 0.8, 12);
    expect(plan('jerky').duration_s).toBeGreaterThan(std.duration_s + 0.17);
    expect(plan('jerky').duration_s).toBeLessThan(std.duration_s + 0.41);
    // curved: larger deviation from the straight line at mid-movement
    const dev = (pl: ReturnType<typeof plan>): number => {
      const m = pl.pose(pl.duration_s / 2).p;
      const lin = [0, 1, 2].map((i) => (pl.p0[i] + pl.p1[i]) / 2);
      return Math.hypot(m[0] - lin[0], m[1] - lin[1], m[2] - lin[2]);
    };
    expect(dev(plan('curved'))).toBeGreaterThan(2.5 * dev(std));
    // pause: a still plateau of 0.8 s in the middle; jerky: two short stops
    const still = (pl: ReturnType<typeof plan>): number => {
      let longest = 0;
      let run = 0;
      for (let t = 0.02; t < pl.duration_s - 0.02; t += 0.005) {
        if (vNorm(pl.pose(t).v) < 1e-6) run += 0.005;
        else run = 0;
        longest = Math.max(longest, run);
      }
      return longest;
    };
    expect(still(std)).toBe(0);
    expect(still(plan('pause'))).toBeGreaterThan(0.75);
    expect(still(plan('pause'))).toBeLessThan(0.85);
    expect(still(plan('jerky'))).toBeGreaterThan(0.07);
    expect(still(plan('jerky'))).toBeLessThan(0.21);
    // twist: more roll, elbow_high: less pitch → different end orientations
    const endTurn = (pl: ReturnType<typeof plan>): number => vNorm(mVecAngle(mMul(pl.pose(pl.duration_s).R, mT(pl.grip))));
    expect(endTurn(plan('twist'))).not.toBeCloseTo(endTurn(std), 3);
    expect(endTurn(plan('elbow_high'))).not.toBeCloseTo(endTurn(std), 3);
  });

  it('the free orientation draws one of the nine grips (mirror for the left hand)', () => {
    const p = setup(8).participant;
    const seen = new Set<string>();
    for (let i = 0; i < 200; i++) {
      const r = planMovement(p, condition({ hand: 'right', orientation: 'free' }), simRng(i));
      const l = planMovement(p, condition({ hand: 'left', orientation: 'free' }), simRng(i));
      expect(GRIP_ORIENTATIONS).toContain(r.orientation);
      expect(l.orientation).toBe(ORIENTATIONS[r.orientation].mirror);
      seen.add(r.orientation);
    }
    expect(seen.size).toBe(9);
  });

  it('the grip tilt never exceeds 10°', () => {
    const p = { ...setup(10).participant };
    p.style = { ...p.style, nondominant_jitter: 2.2 };
    for (let i = 0; i < 300; i++) {
      const hand = i % 2 ? 'left' : 'right';
      const o = GRIP_ORIENTATIONS[i % 9];
      const plan = planMovement(p, condition({ hand, orientation: o }), simRng(i));
      const tilt = vNorm(mVecAngle(mMul(plan.grip, mT(gripMatrix(o)))));
      expect(tilt).toBeLessThanOrEqual(10 + 1e-9);
    }
  });
});

/** rotation angle (deg) as a vector: axis·angle of a rotation matrix */
function mVecAngle(r: Mat3): Vec3 {
  const c = Math.max(-1, Math.min(1, (r[0] + r[4] + r[8] - 1) / 2));
  const ang = Math.acos(c) / DEG;
  return [ang, 0, 0];
}

// ───────────────────────────── participants and platforms ─────────────────────────────

describe('participants and platforms', () => {
  it('simRng is the reference mulberry32', () => {
    // reference implementation (as published by its author)
    function mulberry32(a: number) {
      return function () {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    for (const seed of [0, 1, 42, 0xffffffff, 123456789]) {
      const a = simRng(seed);
      const b = mulberry32(seed);
      for (let i = 0; i < 100; i++) expect(a()).toBe(b());
    }
  });

  it('randomSimParticipant is deterministic and plausible', () => {
    const a = randomSimParticipant(simRng(5));
    expect(randomSimParticipant(simRng(5))).toEqual(a);
    let left = 0;
    for (let i = 0; i < 400; i++) {
      const p = randomSimParticipant(simRng(i));
      expect(p.participant_id).toMatch(/^P-[A-HJ-NP-Z2-9]{6}$/);
      expect(p.age_years).toBeGreaterThanOrEqual(18);
      expect(p.age_years).toBeLessThanOrEqual(80);
      expect(p.height_cm).toBeGreaterThanOrEqual(145);
      expect(p.height_cm).toBeLessThanOrEqual(205);
      expect(p.arm_length_m / (p.height_cm / 100)).toBeGreaterThan(0.32);
      expect(p.arm_length_m / (p.height_cm / 100)).toBeLessThan(0.4);
      expect(p.shoulder_half_width_m).toBeGreaterThan(0.13);
      expect(p.shoulder_half_width_m).toBeLessThan(0.25);
      if (p.handedness !== 'ambidextrous') expect(p.dominant).toBe(p.handedness);
      expect(p.style.nondominant_slowdown).toBeGreaterThanOrEqual(1);
      expect(p.style.tremor_dps).toBeGreaterThan(0.5);
      expect(p.style.tremor_dps).toBeLessThan(2.5);
      if (p.dominant === 'left') left++;
    }
    expect(left / 400).toBeGreaterThan(0.05);
    expect(left / 400).toBeLessThan(0.2);
  });

  it('overrides are honoured and never shift the random draws', () => {
    const a = randomSimParticipant(simRng(11));
    const b = randomSimParticipant(simRng(11), { handedness: 'left', sex: 'female', height_cm: 150 });
    expect(b.handedness).toBe('left');
    expect(b.dominant).toBe('left');
    expect(b.sex).toBe('female');
    expect(b.height_cm).toBe(150);
    expect(b.arm_length_m).toBeLessThan(0.36 * 1.5 * 1.1);
    expect(b.style).toEqual(a.style);
    expect(b.participant_id).toBe(a.participant_id);
  });

  it('simParticipantFromProfile: deterministic, dominance from handedness, arm from height', () => {
    const profile: ParticipantProfile = {
      participant_id: 'P-ABCD23',
      nickname: null,
      age_years: 30,
      sex: 'male',
      handedness_self: 'left',
      ehi_sf: null,
      ehi_lq: null,
      ehi_class: null,
      phone_hand: 'left',
      height_cm: 180,
      arm_issue: false,
      phone_model: null,
      phone_case: null,
      created_at: '2026-10-23T10:00:00Z',
      updated_at: '2026-10-23T10:00:00Z',
    };
    const p = simParticipantFromProfile(profile, 99);
    expect(simParticipantFromProfile(profile, 99)).toEqual(p);
    expect(p).toMatchObject({ participant_id: 'P-ABCD23', sex: 'male', age_years: 30, height_cm: 180, handedness: 'left', dominant: 'left' });
    expect(p.arm_length_m).toBeGreaterThan(0.36 * 1.8 * 0.92);
    expect(p.arm_length_m).toBeLessThan(0.36 * 1.8 * 1.08);
    const doms = new Set<string>();
    for (let seed = 0; seed < 40; seed++) {
      const amb = simParticipantFromProfile({ ...profile, handedness_self: 'ambidextrous', height_cm: null, age_years: null }, seed);
      expect(amb.dominant).toBe(simParticipantFromProfile({ ...profile, handedness_self: 'ambidextrous', height_cm: null, age_years: null }, seed).dominant);
      expect(amb.height_cm).toBeGreaterThan(140);
      doms.add(amb.dominant);
    }
    expect(doms.size).toBe(2); // the coin flip depends on the seed
    const anon = simParticipantFromProfile(null, 3);
    expect(anon.dominant).toBe('right');
    expect(anon.handedness).toBe('right');
  });

  it('defaultSimPlatform: iOS has no Generic Sensor quaternion; gravity sign follows the documented constant', () => {
    const ios = defaultSimPlatform('ios', simRng(1));
    const android = defaultSimPlatform('android', simRng(1));
    expect(ios.quaternion).toBe(false);
    expect(android.quaternion).toBe(true);
    expect(ios.gravitySign).toBe(IOS_GRAVITY_SIGN);
    expect(android.gravitySign).toBe(1);
    for (const p of [ios, android]) {
      expect(p.hz).toBe(60);
      expect(p.acc_available).toBe(true);
      expect(p.jitter_ms).toBeGreaterThan(0);
      expect(p.drop_prob).toBeLessThan(0.03);
      expect(p.heading_deg).toBeGreaterThanOrEqual(0);
      expect(p.heading_deg).toBeLessThan(360);
    }
  });
});

// ───────────────────────────── the source ─────────────────────────────

describe('simulated source (offline mode)', () => {
  it('runs synchronously inside start(), stops when a listener calls stop(), cannot start twice', async () => {
    const st = setup(3);
    const src = offlineSource(st, condition(), 1);
    let nMotion = 0;
    let nOrientationAfterStop = 0;
    let stoppedAt: number | null = null;
    const p = src.start({
      onMotion(s) {
        nMotion++;
        expect(src.clock.now()).toBe(s.t);
        if (nMotion === 50) {
          src.stop();
          stoppedAt = s.t;
        }
      },
      onOrientation(s) {
        if (stoppedAt !== null && s.t > stoppedAt) nOrientationAfterStop++;
      },
    });
    expect(nMotion).toBe(50); // already done before the promise is awaited
    expect(nOrientationAfterStop).toBe(0);
    await expect(p).resolves.toBeUndefined();
    await expect(src.start({ onMotion() {}, onOrientation() {} })).rejects.toThrow();
    expect(src.clock.epochAt(2000)).toBe(1_700_000_000_000 + 2000);
  });

  it('stops by itself after maxDurationMs', async () => {
    const src = offlineSource(setup(4), condition(), 1, { maxDurationMs: 2000 });
    const ts: number[] = [];
    await src.start({ onMotion: (s) => ts.push(s.t), onOrientation() {} });
    expect(ts[ts.length - 1]).toBeLessThanOrEqual(START_T + 2000);
    expect(ts[ts.length - 1]).toBeGreaterThan(START_T + 1900);
    expect(ts[0]).toBeGreaterThan(START_T);
  });

  it('timing: ~60 Hz, strictly increasing t, ts = t − small latency, platform interval units', () => {
    for (const name of ['android', 'ios'] as const) {
      const c = run(12, condition(), { noise: true, platform: name }, { goAtMs: 1000, stopAfterMs: 4000 });
      const m = c.motion;
      for (let i = 1; i < m.length; i++) expect(m[i].t).toBeGreaterThan(m[i - 1].t);
      for (const s of m) {
        expect(s.t - s.ts!).toBeGreaterThanOrEqual(0.3);
        expect(s.t - s.ts!).toBeLessThanOrEqual(2.0);
        expect(s.interval).toBe(name === 'ios' ? Math.fround(1 / 60) : 16);
      }
      const span = m[m.length - 1].t - m[0].t;
      const hz = ((m.length - 1) / span) * 1000;
      expect(hz).toBeGreaterThan(55);
      expect(hz).toBeLessThan(61);
      // orientation events right after each motion event (the last one is skipped when the
      // listener stops the source on the final motion sample)
      expect(m.length - c.orientation.length).toBeGreaterThanOrEqual(0);
      expect(m.length - c.orientation.length).toBeLessThanOrEqual(1);
      for (let i = 0; i < c.orientation.length; i++) {
        expect(c.orientation[i].t - m[i].t).toBeGreaterThanOrEqual(0.2);
        expect(c.orientation[i].t - m[i].t).toBeLessThanOrEqual(1.5);
        expect(c.orientation[i].absolute).toBe(false);
      }
      if (name === 'ios') expect(c.quaternion.length).toBe(0);
      else {
        expect(c.quaternion.length).toBeGreaterThan(m.length * 0.9);
        for (const q of c.quaternion) {
          expect(q.sensor).toBe('RelativeOrientationSensor');
          expect(Math.hypot(...q.q)).toBeCloseTo(1, 12);
        }
        for (let i = 1; i < c.quaternion.length; i++) expect(c.quaternion[i].t).toBeGreaterThan(c.quaternion[i - 1].t);
      }
    }
  });

  it('jitter and drops follow the platform parameters; noise:false is perfectly regular', () => {
    const st = setup(13);
    const jittery = { ...st, platform: { ...st.platform, drop_prob: 0.2, jitter_ms: 3 } };
    const c = collect(offlineSource(jittery, condition(), 2, { behavior: 'no_move' }), { goAtMs: null, stopAfterMs: 10000 });
    const nominal = 10000 / (1000 / 60);
    expect(c.motion.length).toBeGreaterThan(nominal * 0.6);
    expect(c.motion.length).toBeLessThan(nominal * 0.85);
    const dts = c.motion.slice(1).map((s, i) => s.t - c.motion[i].t);
    expect(Math.max(...dts)).toBeGreaterThan(50); // stalls produce gaps
    const clean = run(13, condition(), { noise: false }, { goAtMs: null, stopAfterMs: 2000 });
    const cdts = clean.motion.slice(1).map((s, i) => s.t - clean.motion[i].t);
    for (const dt of cdts) expect(dt).toBeCloseTo(1000 / 60, 9);
  });

  it('platform switches: gravitySign multiplies acc/accg, acc_available false gives null acc', () => {
    const st = setup(14);
    const c = condition({ orientation: 'top_left_toward' });
    const co = { goAtMs: 1000, stopAfterMs: 3000 };
    const pos = collect(offlineSource(st, c, 7, { noise: false }), co);
    const neg = collect(offlineSource({ ...st, platform: { ...st.platform, gravitySign: -1 } }, c, 7, { noise: false }), co);
    const noAcc = collect(offlineSource({ ...st, platform: { ...st.platform, acc_available: false } }, c, 7, { noise: false }), co);
    expect(neg.motion.length).toBe(pos.motion.length);
    for (let i = 0; i < pos.motion.length; i++) {
      for (let a = 0; a < 3; a++) {
        // x + (−x) is +0 for every x (avoids ±0 distinctions of toBe)
        expect(neg.motion[i].accg![a] + pos.motion[i].accg![a]).toBe(0);
        expect(neg.motion[i].acc![a] + pos.motion[i].acc![a]).toBe(0);
        expect(noAcc.motion[i].accg![a]).toBe(pos.motion[i].accg![a]);
      }
      expect(neg.motion[i].rr).toEqual(pos.motion[i].rr);
      expect(noAcc.motion[i].acc).toBeNull();
    }
  });

  it('orientation, quaternion, accg and rotationRate are mutually consistent (W3C conventions)', () => {
    const c = run(15, condition({ orientation: 'top_right_away', variant: 'twist' }), { noise: false }, { goAtMs: 1000, stopAfterMs: 3500 });
    const m = c.motion;
    // gravity predicted from the Euler angles equals accg at rest (Rᵀ·up, W3C sign)
    const o0 = c.orientation[0];
    const R0 = matFromEulerZXY(o0.alpha!, o0.beta!, o0.gamma!);
    const g0 = mVec(mT(R0), [0, 0, GRAVITY]);
    for (let a = 0; a < 3; a++) expect(m[0].accg![a]).toBeCloseTo(g0[a], 6);
    // quaternion and Euler describe the same rotation (at rest, where timing offsets do not matter)
    const q0 = c.quaternion.find((q) => q.t < c.source.truth.onset_t!)!;
    const Rq = matFromQuat(q0.q);
    for (let i = 0; i < 9; i++) expect(Rq[i]).toBeCloseTo(R0[i], 9);
    // integrating rotationRate (device frame) over the movement reproduces the orientation change
    const i0 = m.findIndex((s) => s.ts! > c.source.truth.onset_t! - 100);
    const i1 = m.findIndex((s) => s.ts! > c.source.truth.end_t! + 100);
    let R = matFromEulerZXY(c.orientation[i0].alpha!, c.orientation[i0].beta!, c.orientation[i0].gamma!);
    for (let i = i0; i < i1; i++) {
      const w0 = rotationRateToVector(m[i].rr!);
      const w1 = rotationRateToVector(m[i + 1].rr!);
      const dt = (m[i + 1].ts! - m[i].ts!) / 1000;
      R = mMul(R, mExp([((w0[0] + w1[0]) / 2) * DEG * dt, ((w0[1] + w1[1]) / 2) * DEG * dt, ((w0[2] + w1[2]) / 2) * DEG * dt]));
    }
    const oe = c.orientation[i1];
    const Re = matFromEulerZXY(oe.alpha!, oe.beta!, oe.gamma!);
    const err = vNorm(mVecAngle(mMul(R, mT(Re))));
    const turn = vNorm(mVecAngle(mMul(Re, mT(matFromEulerZXY(c.orientation[i0].alpha!, c.orientation[i0].beta!, c.orientation[i0].gamma!)))));
    expect(turn).toBeGreaterThan(20);
    expect(err).toBeLessThan(0.02 * turn + 0.5);
  });
});

// ───────────────────────────── behaviours ─────────────────────────────

describe('behaviours', () => {
  it('holding still before GO passes the default hold-still criterion (all participants, both hands and platforms)', () => {
    let worstG = 0;
    let worstA = 0;
    for (let seed = 0; seed < 30; seed++) {
      for (const platform of ['android', 'ios'] as const) {
        const st = setup(2000 + seed, platform);
        for (const hand of ['right', 'left'] as const) {
          const c = collect(offlineSource(st, condition({ hand, orientation: GRIP_ORIENTATIONS[seed % 9] }), seed), { goAtMs: 2500, stopAfterMs: 200 });
          const sm = smoothed(c.motion.filter((s) => s.t < c.source.truth.go_t!));
          worstG = Math.max(worstG, ...sm.gyro);
          worstA = Math.max(worstA, ...sm.acc);
        }
      }
    }
    expect(worstG).toBeLessThan(0.6 * D.stillGyro);
    expect(worstA).toBeLessThan(0.75 * D.stillAcc);
    expect(worstG).toBeGreaterThan(0.5); // tremor is there
  });

  it('normal: onset = GO + reaction (±25 %), then a clearly detectable movement', () => {
    for (let seed = 0; seed < 10; seed++) {
      const st = setup(3000 + seed);
      const c = collect(offlineSource(st, condition(), seed), { goAtMs: 1500, stopAfterMs: 4000 });
      const tr = c.source.truth;
      expect(tr.behavior).toBe('normal');
      const reaction = tr.onset_t! - tr.go_t!;
      expect(reaction).toBeGreaterThanOrEqual(0.75 * st.participant.style.reaction_ms - 1e-9);
      expect(reaction).toBeLessThanOrEqual(1.25 * st.participant.style.reaction_ms + 1e-9);
      const sm = smoothed(c.motion);
      const during = sm.gyro.filter((_, i) => sm.t[i] > tr.onset_t! && sm.t[i] < tr.end_t!);
      expect(Math.max(...during)).toBeGreaterThan(D.onsetGyro);
    }
  });

  it(`false_start: an abrupt twitch ${FALSE_START_AFTER_STILL_MS} ms after start that the segmenter would classify as a false start`, () => {
    for (let seed = 0; seed < 12; seed++) {
      const st = setup(4000 + seed, seed % 2 ? 'ios' : 'android');
      const c = collect(offlineSource(st, condition({ hand: seed % 3 ? 'right' : 'left' }), seed, { behavior: 'false_start' }), { goAtMs: 2600, stopAfterMs: 3000 });
      const tr = c.source.truth;
      expect(tr.false_start_t).toBe(START_T + FALSE_START_AFTER_STILL_MS);
      const sm = smoothed(c.motion);
      // first sample that leaves the still region decides false_start vs still_lost
      const first = sm.t.findIndex((_, i) => sm.gyro[i] >= D.stillGyro || sm.acc[i] >= D.stillAcc);
      expect(sm.t[first]).toBeGreaterThan(tr.false_start_t!);
      expect(sm.t[first]).toBeLessThan(tr.false_start_t! + 40);
      expect(sm.gyro[first] > D.onsetGyro || sm.acc[first] > D.onsetAcc).toBe(true);
      // still again ~300 ms after the twitch and until GO
      for (let i = 0; i < sm.t.length; i++) {
        if (sm.t[i] > tr.false_start_t! + 250 + 300 && sm.t[i] < tr.go_t!) {
          expect(sm.gyro[i]).toBeLessThan(D.stillGyro);
          expect(sm.acc[i]).toBeLessThan(D.stillAcc);
        }
      }
      // then a normal movement after GO
      expect(tr.onset_t).not.toBeNull();
      expect(Math.max(...sm.gyro.filter((_, i) => sm.t[i] > tr.onset_t!))).toBeGreaterThan(D.onsetGyro);
    }
  });

  it('false_start is cancelled when GO comes first', () => {
    const c = collect(offlineSource(setup(5), condition(), 1, { behavior: 'false_start' }), { goAtMs: 500, stopAfterMs: 3000 });
    expect(c.source.truth.false_start_t).toBeNull();
    const sm = smoothed(c.motion.filter((s) => s.t < c.source.truth.onset_t!));
    expect(Math.max(...sm.gyro)).toBeLessThan(D.stillGyro);
  });

  it('no_move ignores GO and stays still', () => {
    const c = collect(offlineSource(setup(6), condition(), 1, { behavior: 'no_move' }), { goAtMs: 1000, stopAfterMs: 7000 });
    const tr = c.source.truth;
    expect(tr.go_t).not.toBeNull();
    expect(tr.onset_t).toBeNull();
    expect(tr.end_t).toBeNull();
    const sm = smoothed(c.motion);
    expect(Math.max(...sm.gyro)).toBeLessThan(D.stillGyro);
    expect(Math.max(...sm.acc)).toBeLessThan(D.stillAcc);
  });

  it('too_slow: a detectable onset, then movement that never settles for 20 s', () => {
    for (const speed of ['natural', 'slow'] as const) {
      const c = collect(offlineSource(setup(7), condition({ speed }), 1, { behavior: 'too_slow' }), { goAtMs: 1000, stopAfterMs: 13000 });
      const tr = c.source.truth;
      expect(tr.end_t! - tr.onset_t!).toBe(20000);
      const sm = smoothed(c.motion);
      const firstOn = sm.t.findIndex((t, i) => t > tr.onset_t! && sm.gyro[i] > D.onsetGyro);
      expect(sm.t[firstOn] - tr.onset_t!).toBeLessThan(600);
      // never 300 ms continuously below the end thresholds (the strictest: slow speed 10 deg/s)
      let run = 0;
      let longest = 0;
      for (let i = 1; i < sm.t.length; i++) {
        if (sm.t[i] < tr.onset_t! + 600 || sm.t[i] > tr.onset_t! + 12000) continue;
        if (sm.gyro[i] < 10 && sm.acc[i] < 0.25) run += sm.t[i] - sm.t[i - 1];
        else run = 0;
        longest = Math.max(longest, run);
      }
      expect(longest).toBeLessThan(100);
    }
  });

  it('static: phone flat on a table, no tremor, only sensor noise and bias', () => {
    const st = setup(8);
    const c = collect(offlineSource(st, condition({ orientation: 'flat_up' }), 1, { behavior: 'static' }), { goAtMs: 500, stopAfterMs: 3000 });
    expect(c.source.truth.onset_t).toBeNull();
    const mean = [0, 1, 2].map((a) => c.motion.reduce((s, x) => s + x.accg![a], 0) / c.motion.length) as Vec3;
    expect(angleBetweenDeg(mean, [0, 0, 1])).toBeLessThan(0.2);
    expect(vNorm(mean)).toBeCloseTo(GRAVITY, 1);
    const rr = [0, 1, 2].map((a) => c.motion.reduce((s, x) => s + rotationRateToVector(x.rr!)[a], 0) / c.motion.length);
    for (let a = 0; a < 3; a++) expect(rr[a]).toBeCloseTo(st.platform.gyro_bias_dps[a], 1);
    // exactly the nominal grip without noise
    const clean = collect(offlineSource(st, condition({ orientation: 'flat_up' }), 1, { behavior: 'static', noise: false }), { goAtMs: 500, stopAfterMs: 1000 });
    for (const s of clean.motion) expect(s.accg).toEqual([0, 0, GRAVITY]);
  });

  it('end cue: the arm returns along the path, starting 250–450 ms later (never before arrival)', () => {
    const c = run(9, condition(), { noise: false }, { goAtMs: 1000, endCueAfterArriveMs: 300, stopAfterMs: 5000 });
    const tr = c.source.truth;
    expect(tr.return_t! - (tr.end_t! + 300)).toBeGreaterThanOrEqual(250 - 20);
    expect(tr.return_t! - (tr.end_t! + 300)).toBeLessThanOrEqual(450 + 20);
    const before = c.motion.filter((s) => s.ts! > tr.end_t! + 70 && s.ts! < tr.return_t!);
    expect(Math.max(...before.map(gyroMag))).toBeLessThan(1e-6);
    const returning = c.motion.filter((s) => s.ts! > tr.return_t! && s.ts! < tr.return_t! + 800);
    expect(Math.max(...returning.map(gyroMag))).toBeGreaterThan(30);
    const first = c.motion[0].accg!;
    const lastS = c.motion[c.motion.length - 1].accg!;
    for (let a = 0; a < 3; a++) expect(lastS[a]).toBeCloseTo(first[a], 9); // back at the start pose
  });
});

// ───────────────────────────── protocol 2.0.0 (street) behaviours ─────────────────────────────

describe('protocol 2.0.0: the return of the arm, starting at the nose, sway, tilted touch, choice trial', () => {
  /** Rotation angle (deg) of R. */
  const angleOf = (R: Mat3): number => (vNorm(mLog(R)) * 180) / Math.PI;

  it('every trial starts still with the arm out (the grip pose), not at the nose', () => {
    for (let seed = 0; seed < 6; seed++) {
      const c = collect(offlineSource(setup(9500 + seed), condition({ orientation: 'top_left_toward' }), seed, { noise: false }), {
        goAtMs: 800,
        stopAfterMs: 100,
      });
      const plan = c.source.plan;
      for (const s of c.motion.filter((x) => x.ts! < c.source.truth.go_t!)) {
        expect(gyroMag(s)).toBeLessThan(1e-9);
        expect(angleBetweenDeg(s.accg!, mVec(mT(plan.grip), [0, 1, 0]))).toBeLessThan(1e-6);
      }
    }
  });

  it('end cue: the return starts 250–450 ms later, lasts 0.8–1.2 s and ends still at the start pose', () => {
    const delays: number[] = [];
    const durations: number[] = [];
    for (let seed = 0; seed < 60; seed++) {
      const c = run(9600 + seed, condition({ speed: (['natural', 'fast', 'slow'] as const)[seed % 3] }), { noise: false }, {
        goAtMs: 1000,
        endCueAfterArriveMs: 300,
        stopAfterMs: 7000,
      });
      const tr = c.source.truth;
      const cueT = c.motion.find((s) => s.t >= tr.end_t! + 300)!.t; // the motion sample that sent the cue
      delays.push(tr.return_t! - cueT);
      durations.push(tr.return_end_t! - tr.return_t!);
      const after = c.motion.filter((s) => s.ts! > tr.return_end_t! + 5);
      expect(after.length).toBeGreaterThan(10);
      for (const s of after) {
        expect(gyroMag(s)).toBeLessThan(1e-6);
        for (let a = 0; a < 3; a++) expect(s.accg![a]).toBeCloseTo(c.motion[0].accg![a], 6);
      }
      const back = c.motion.filter((s) => s.ts! > tr.return_t! && s.ts! < tr.return_end_t!);
      expect(Math.max(...back.map(gyroMag))).toBeGreaterThan(45); // a return the controller can see
    }
    expect(Math.min(...delays)).toBeGreaterThanOrEqual(250);
    expect(Math.max(...delays)).toBeLessThanOrEqual(450);
    expect(Math.max(...delays) - Math.min(...delays)).toBeGreaterThan(120); // drawn, not fixed
    expect(Math.min(...durations)).toBeGreaterThanOrEqual(800);
    expect(Math.max(...durations)).toBeLessThanOrEqual(1200);
    expect(Math.max(...durations) - Math.min(...durations)).toBeGreaterThan(200);
  });

  it('returnDelayMs: a participant who lingers at the nose returns that long after the cue', () => {
    const c = collect(offlineSource(setup(9700), condition(), 3, { noise: false, returnDelayMs: 3500 }), {
      goAtMs: 1000,
      endCueAfterArriveMs: 300,
      stopAfterMs: 9000,
    });
    const tr = c.source.truth;
    const cueT = c.motion.find((s) => s.t >= tr.end_t! + 300)!.t;
    expect(tr.return_t! - cueT).toBe(3500);
    expect(maxOver(c.motion, gyroMag, tr.end_t! + 100, tr.return_t!)).toBeLessThan(1e-6);
  });

  it('startAtNose: still at the nose pose, back out 300–800 ms after start (0.8–1.2 s), then still at p0 until GO', () => {
    for (let seed = 0; seed < 8; seed++) {
      const st = setup(9800 + seed);
      const c = collect(offlineSource(st, condition({ speed: 'fast' }), seed, { noise: false, startAtNose: true }), { goAtMs: 3500, stopAfterMs: 100 });
      const tr = c.source.truth;
      const plan = c.source.plan;
      const delay = tr.initial_return_t! - START_T;
      expect(delay).toBeGreaterThanOrEqual(300);
      expect(delay).toBeLessThanOrEqual(800);
      const atNose = mVec(mT(plan.pose(plan.duration_s).R), [0, 1, 0]);
      const out = mVec(mT(plan.grip), [0, 1, 0]);
      for (const s of c.motion) {
        if (s.ts! < tr.initial_return_t!) expect(angleBetweenDeg(s.accg!, atNose)).toBeLessThan(1e-6);
      }
      // the return ends somewhere in 0.8–1.2 s; from then on the phone is still at the start pose
      const settled = c.motion.filter((s) => s.ts! > tr.initial_return_t! + 1200 && s.ts! < tr.go_t!);
      expect(settled.length).toBeGreaterThan(30);
      for (const s of settled) {
        expect(angleBetweenDeg(s.accg!, out)).toBeLessThan(1e-6);
        expect(gyroMag(s)).toBeLessThan(1e-6);
      }
      expect(maxOver(c.motion, gyroMag, tr.initial_return_t!, tr.initial_return_t! + 1200)).toBeGreaterThan(45);
      expect(tr.onset_t).not.toBeNull(); // then the trial's own movement after GO
    }
    // an anticipation or a sway comes once the arm is out again, not during the return
    const fs = offlineSource(setup(9900), condition(), 1, { behavior: 'false_start', startAtNose: true });
    collect(fs, { goAtMs: 5000, stopAfterMs: 100 });
    expect(fs.truth.false_start_t! - fs.truth.initial_return_t!).toBeGreaterThan(800 + FALSE_START_AFTER_STILL_MS - 1);
  });

  it("sway: a slow posture shift before GO breaks a 20 deg/s hold but stays under the 30 deg/s / 0.6 m/s² false-start floors", () => {
    let maxG = 0;
    let minG = Infinity;
    let maxA = 0;
    for (let seed = 0; seed < 20; seed++) {
      for (const platform of ['android', 'ios'] as const) {
        const st = setup(10_000 + seed, platform);
        for (const hand of ['right', 'left'] as const) {
          const c = collect(offlineSource(st, condition({ hand }), seed, { behavior: 'sway' }), { goAtMs: 2500, stopAfterMs: 100 });
          const tr = c.source.truth;
          expect(tr.sway_t).toBe(START_T + FALSE_START_AFTER_STILL_MS);
          const sm = smoothed(c.motion);
          let g = 0;
          let a = 0;
          for (let i = 0; i < sm.t.length; i++) {
            if (sm.t[i] < tr.sway_t! || sm.t[i] > tr.sway_t! + 700) continue;
            g = Math.max(g, sm.gyro[i]);
            a = Math.max(a, sm.acc[i]);
          }
          maxG = Math.max(maxG, g);
          minG = Math.min(minG, g);
          maxA = Math.max(maxA, a);
        }
      }
    }
    expect(minG).toBeGreaterThan(20);
    expect(maxG).toBeLessThan(30);
    expect(maxA).toBeLessThan(0.6);
    // GO before the sway cancels it
    const early = offlineSource(setup(10_100), condition(), 1, { behavior: 'sway' });
    collect(early, { goAtMs: 300, stopAfterMs: 3000 });
    expect(early.truth.sway_t).toBeNull();
    expect(early.truth.onset_t).not.toBeNull();
    // anticipationAfterMs moves the anticipation (e.g. into the longer foreperiod of other parameters)
    for (const behavior of ['sway', 'false_start'] as const) {
      const late = offlineSource(setup(10_101), condition(), 1, { behavior, anticipationAfterMs: 900 });
      collect(late, { goAtMs: 2500, stopAfterMs: 100 });
      expect(behavior === 'sway' ? late.truth.sway_t : late.truth.false_start_t).toBe(START_T + 900);
    }
  });

  it('tilted_touch: the same movement, with the end pose turned 30–40° about the device x axis', () => {
    const angles: number[] = [];
    for (let seed = 0; seed < 25; seed++) {
      const st = setup(10_200 + seed);
      const hand = seed % 2 ? 'left' : 'right';
      const tilted = offlineSource(st, condition({ hand, variant: 'tilted_touch' }), seed);
      const standard = offlineSource(st, condition({ hand, variant: 'standard' }), seed);
      expect(tilted.plan.duration_s).toBe(standard.plan.duration_s);
      expectMatClose(tilted.plan.pose(0).R, standard.plan.pose(0).R); // same start
      const d = mLog(mMul(mT(standard.plan.pose(standard.plan.duration_s).R), tilted.plan.pose(tilted.plan.duration_s).R));
      expect(Math.abs(d[1])).toBeLessThan(1e-9);
      expect(Math.abs(d[2])).toBeLessThan(1e-9);
      angles.push((d[0] * 180) / Math.PI);
    }
    expect(Math.min(...angles)).toBeGreaterThanOrEqual(30);
    expect(Math.max(...angles)).toBeLessThanOrEqual(40);
    // a direction, not a sign convention: the top edge tips towards the face (+x rotation for both hands)
    expect(angles.every((a) => a > 0)).toBe(true);
    expect(angleOf(mIdentity())).toBe(0);
  });

  it("choice trial (hand_source 'reported'): the participant's own phone hand ('both' → dominant) and the upright grip", () => {
    const st = setup(10_300);
    const p = { ...st.participant, dominant: 'right' as const };
    const choice = condition({ hand: 'right', orientation: 'free', hand_source: 'reported' });
    expect(performedCondition({ ...p, phone_hand: 'left' }, choice)).toEqual({ ...choice, hand: 'left', orientation: 'upright_toward' });
    expect(performedCondition({ ...p, phone_hand: 'both' }, choice)).toMatchObject({ hand: 'right', orientation: 'upright_toward' });
    expect(performedCondition(p, choice).hand).toBe('right'); // unknown phone hand → dominant
    expect(simChoiceHand({ ...p, dominant: 'left' })).toBe('left');
    // instructed trials (and v1 'free' blocks) are performed exactly as planned
    for (const c of [condition({ hand: 'left' }), condition({ orientation: 'free' }), condition({ hand_source: 'instructed' })]) {
      expect(performedCondition({ ...p, phone_hand: 'left' }, c)).toBe(c);
    }
    // the simulated source performs it
    const src = offlineSource({ ...st, participant: { ...p, phone_hand: 'left' } }, choice, 4, { noise: false });
    expect(src.performed).toMatchObject({ hand: 'left', orientation: 'upright_toward', hand_source: 'reported' });
    expect(src.plan.h).toBe(-1);
    expectMatClose(src.plan.grip, gripMatrix('upright_toward'), 0.2); // the nominal upright grip (± the ≤ 10° tilt)
    // simParticipantFromProfile copies the usual phone hand
    const profile = { participant_id: 'P-ABCDEF', sex: 'female', handedness_self: 'right', phone_hand: 'left', age_years: 30, height_cm: 170 } as ParticipantProfile;
    expect(simParticipantFromProfile(profile, 5).phone_hand).toBe('left');
  });

  it('absolute orientation streams (Android-like platforms): one per relative sample, same pose up to fusion noise; none on iOS', () => {
    const got = (platform: 'ios' | 'android') => {
      const src = offlineSource(setup(10_400, platform), condition(), 2, { behavior: 'no_move', maxDurationMs: 1500 });
      const rel: QuaternionSample[] = [];
      const abs: QuaternionSample[] = [];
      const ori: OrientationSample[] = [];
      const oriAbs: OrientationSample[] = [];
      void src.start({
        onMotion() {},
        onOrientation: (s) => ori.push(s),
        onQuaternion: (s) => rel.push(s),
        onQuaternionAbs: (s) => abs.push(s),
        onOrientationAbs: (s) => oriAbs.push(s),
      });
      return { rel, abs, ori, oriAbs };
    };
    const a = got('android');
    expect(a.abs.length).toBe(a.rel.length);
    expect(a.oriAbs.length).toBe(a.ori.length);
    expect(a.abs.length).toBeGreaterThan(60);
    a.abs.forEach((s, i) => {
      expect(s.sensor).toBe('AbsoluteOrientationSensor');
      expect(s.t).toBe(a.rel[i].t);
      const d = mMul(mT(matFromQuat(s.q)), matFromQuat(a.rel[i].q));
      expect(angleOf(d)).toBeLessThan(1); // both fused from the same pose
    });
    a.oriAbs.forEach((s, i) => {
      expect(s.absolute).toBe(true);
      expect(a.ori[i].absolute).toBe(false);
      expect(s.ts).toBe(a.ori[i].ts);
    });
    const i = got('ios');
    expect(i.abs).toEqual([]);
    expect(i.oriAbs).toEqual([]);
    expect(i.ori.length).toBeGreaterThan(60);
  });
});

// ───────────────────────────── realtime mode ─────────────────────────────

describe('simulated source (realtime mode)', () => {
  it('emits samples on a timer with performance.now() timestamps until stopped', async () => {
    const st = setup(16);
    const src = createSimulatedSource({ mode: 'realtime', participant: st.participant, platform: st.platform, condition: condition(), seed: 1 });
    const got: MotionSample[] = [];
    let quats = 0;
    const t0 = performance.now();
    await src.start({
      onMotion(s) {
        got.push(s);
        if (got.length === 3) src.notifyCue?.('go', s.t);
      },
      onOrientation() {},
      onQuaternion() {
        quats++;
      },
    });
    await new Promise((r) => setTimeout(r, 350));
    src.stop();
    const n = got.length;
    await new Promise((r) => setTimeout(r, 100));
    expect(got.length).toBe(n); // nothing after stop()
    expect(n).toBeGreaterThan(8);
    expect(quats).toBeGreaterThan(5);
    expect(got[0].t).toBeGreaterThanOrEqual(t0);
    expect(got[n - 1].t).toBeLessThanOrEqual(performance.now());
    for (let i = 1; i < n; i++) expect(got[i].t).toBeGreaterThan(got[i - 1].t);
    expect(src.truth.go_t).toBe(got[2].t);
    expect(Math.abs(src.clock.now() - performance.now())).toBeLessThan(50);
    expect(src.clock.epochAt(100)).toBeCloseTo(performance.timeOrigin + 100, 6);
  });
});
