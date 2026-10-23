import { beforeAll, describe, expect, it, vi } from 'vitest';

// sessionFlow is imported only for its pure helpers (trialSeed, reconcileProgress); the services
// module it imports would pull in IndexedDB / Web Audio, so it is replaced by an inert stub.
vi.mock('../src/app/services', () => ({
  getServices: () => {
    throw new Error('services are not used by the synth tests');
  },
  DEFAULT_CONDITION: {
    hand: 'right',
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
  },
}));

import { reconcileProgress, trialSeed } from '../src/app/sessionFlow';
import { GRAVITY } from '../src/core/constants';
import { flattenPlan, normalizeCustomSpec } from '../src/core/protocol';
import type { MotionSample, OrientationSample, QuaternionSample, TrialRecord } from '../src/core/types';
import { makeRng, scoreEhi } from '../src/core/util';
import {
  appTrialSeed,
  generateDataset,
  planDatasetDesign,
  resolveDatasetOptions,
  seededUuid,
  summarizeSession,
  type SyntheticDataset,
} from '../src/sim/dataset';
import { IOS_GRAVITY_SIGN, defaultSimPlatform, simRng } from '../src/sim/model';
import { CHROMIUM_ROUNDING, roundQuaternionLikeChromium, roundToMultiple } from '../src/sim/simulatedSource';
import { condition, offlineSource, setup } from './helpers/simHarness';
import { expectEqualColumnLengths, expectValid } from './helpers/schema';

const SMALL = { participants: 2, preset: 'quick', seed: 2026, failureRate: 0.2 } as const;

/** |v − k·step| for the nearest integer k (0 when v is on the grid). */
function offGrid(v: number, step: number): number {
  return Math.abs(v - Math.round(v / step) * step);
}

function streamValues(t: TrialRecord, cols: readonly (keyof TrialRecord['streams']['motion'])[]): number[] {
  const out: number[] = [];
  for (const c of cols) for (const v of t.streams.motion[c] as (number | null)[]) if (v !== null) out.push(v);
  return out;
}

function nearRound(values: number[], step: number, tol: number): number {
  return values.filter((v) => offGrid(v, step) < tol).length / Math.max(1, values.length);
}

let ds: SyntheticDataset;
beforeAll(async () => {
  ds = await generateDataset(SMALL);
});

// ───────────────────────────── Chromium quantisation ─────────────────────────────

describe('simulator: Chromium-style quantisation (platform.quantize)', () => {
  it('defaultSimPlatform: android quantises, ios does not, and the other draws are unchanged', () => {
    const a = defaultSimPlatform('android', simRng(5));
    const i = defaultSimPlatform('ios', simRng(5));
    expect(a.quantize).toBe(true);
    expect(i.quantize).toBe(false);
    // the flag consumes no random number: the rest of the platform is what it was before
    expect(a.heading_deg).toBe(defaultSimPlatform('android', simRng(5)).heading_deg);
    expect(i.gravitySign).toBe(IOS_GRAVITY_SIGN);
  });

  it('roundToMultiple is Chromium RoundToMultiple: nearest multiple, halves away from zero, symmetric', () => {
    expect(roundToMultiple(1.24, 0.1)).toBeCloseTo(1.2, 12);
    expect(roundToMultiple(1.25, 0.1)).toBeCloseTo(1.3, 12);
    expect(roundToMultiple(-1.24, 0.1)).toBeCloseTo(-1.2, 12);
    expect(roundToMultiple(-1.25, 0.1)).toBeCloseTo(-1.3, 12);
    const rng = makeRng(3);
    for (let k = 0; k < 2000; k++) {
      const v = (rng() - 0.5) * 40;
      expect(roundToMultiple(-v, 0.1)).toBe(-roundToMultiple(v, 0.1)); // mirror (x → −x) commutes
      expect(Math.abs(roundToMultiple(v, 0.1) - v)).toBeLessThanOrEqual(0.05 + 1e-12);
    }
  });

  it('quaternion rounding keeps unit length and moves the rotation by at most ~0.1°', () => {
    const rng = makeRng(4);
    for (let k = 0; k < 500; k++) {
      const raw = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng()];
      const n = Math.hypot(...raw);
      const q = raw.map((x) => x / n) as [number, number, number, number];
      const r = roundQuaternionLikeChromium(q);
      expect(Math.hypot(...r)).toBeCloseTo(1, 12);
      const dot = Math.abs(q[0] * r[0] + q[1] * r[1] + q[2] * r[2] + q[3] * r[3]);
      const angleDeg = (2 * Math.acos(Math.min(1, dot)) * 180) / Math.PI;
      expect(angleDeg).toBeLessThan(0.3);
    }
    expect(roundQuaternionLikeChromium([0, 0, 0, 1])).toEqual([0, 0, 0, 1]);
  });

  function collectAll(platform: 'ios' | 'android', noise: boolean) {
    const st = setup(41, platform);
    const src = offlineSource(st, condition({ orientation: 'top_left_away' }), 3, { noise, behavior: 'no_move', maxDurationMs: 1500 });
    const motion: MotionSample[] = [];
    const orientation: OrientationSample[] = [];
    const quaternion: QuaternionSample[] = [];
    void src.start({ onMotion: (s) => motion.push(s), onOrientation: (s) => orientation.push(s), onQuaternion: (s) => quaternion.push(s) });
    return { motion, orientation, quaternion };
  }

  it('android with noise: acc/accg on the 0.1 m/s² grid, rotationRate on 0.1 deg/s, angles on 0.1°', () => {
    const c = collectAll('android', true);
    expect(c.motion.length).toBeGreaterThan(50);
    const gyroStepDeg = (CHROMIUM_ROUNDING.gyro_rad_s * 180) / Math.PI;
    for (const s of c.motion) {
      for (let a = 0; a < 3; a++) {
        expect(offGrid(s.accg![a], 0.1)).toBeLessThan(1e-9);
        expect(offGrid(s.acc![a], 0.1)).toBeLessThan(1e-9);
      }
      for (const v of [s.rr!.alpha, s.rr!.beta, s.rr!.gamma]) expect(offGrid(v, gyroStepDeg)).toBeLessThan(1e-9);
    }
    for (const o of c.orientation) for (const v of [o.alpha!, o.beta!, o.gamma!]) expect(offGrid(v, 0.1)).toBeLessThan(1e-9);
    expect(c.quaternion.length).toBeGreaterThan(50);
    for (const q of c.quaternion) expect(Math.hypot(...q.q)).toBeCloseTo(1, 12);
  });

  it('ios keeps full precision, and noise:false switches the quantisation off (exact signals)', () => {
    const ios = collectAll('ios', true);
    const accg = ios.motion.flatMap((s) => [...s.accg!]);
    expect(nearRound(accg, 0.1, 1e-6)).toBeLessThan(0.05);
    const exact = collectAll('android', false);
    for (const s of exact.motion) expect(Math.hypot(...s.accg!)).toBeCloseTo(GRAVITY, 9);
    expect(nearRound(exact.motion.flatMap((s) => [...s.accg!]), 0.1, 1e-6)).toBeLessThan(0.5);
  });
});

// ───────────────────────────── design ─────────────────────────────

describe('generateDataset: participant design', () => {
  it('left-handed fraction = round(N · f), spread over both platforms; platforms alternate; sexes balanced', () => {
    for (const [n, f] of [
      [2, 0.3],
      [10, 0.3],
      [12, 0.3],
      [7, 0.5],
      [9, 0],
      [5, 1],
    ] as const) {
      const d = planDatasetDesign({ participants: n, leftFraction: f, seed: 11 });
      const left = d.filter((p) => p.dominant === 'left');
      expect(left.length, `${n} × ${f}`).toBe(Math.round(n * f));
      d.forEach((p, i) => expect(p.platform).toBe(i % 2 === 0 ? 'ios' : 'android'));
      const iosLeft = left.filter((p) => p.platform === 'ios').length;
      expect(Math.abs(iosLeft - (left.length - iosLeft)), `${n} × ${f}`).toBeLessThanOrEqual(1);
      const females = d.filter((p) => p.sex === 'female').length;
      expect(Math.abs(females - (n - females))).toBeLessThanOrEqual(1);
      // first hand crossed with platform: parity flips every two participants
      d.forEach((p, i) => expect(p.seedParity).toBe(Math.floor(i / 2) % 2));
    }
  });

  it('rejects options that would produce invalid records', () => {
    expect(() => resolveDatasetOptions({ participants: 0 })).toThrow();
    expect(() => resolveDatasetOptions({ leftFraction: 1.5 })).toThrow();
    expect(() => resolveDatasetOptions({ failureRate: 1 })).toThrow();
    expect(() => resolveDatasetOptions({ studyCode: 'bad code!' })).toThrow();
    expect(() => resolveDatasetOptions({ iosGravitySign: 0 as unknown as 1 })).toThrow();
    expect(resolveDatasetOptions().studyCode).toBe('synthetic-v1');
  });

  it('appTrialSeed is identical to app/sessionFlow.trialSeed (re-simulation of an attempt)', () => {
    const rng = makeRng(9);
    for (let k = 0; k < 300; k++) {
      const seed = Math.floor(rng() * 4294967296) >>> 0;
      const idx = Math.floor(rng() * 200);
      const attempt = 1 + Math.floor(rng() * 6);
      expect(appTrialSeed(seed, idx, attempt)).toBe(trialSeed(seed, idx, attempt));
    }
  });

  it('seededUuid has the util.newId layout (schema Id) and is reproducible', () => {
    const a = seededUuid(makeRng(1));
    expect(a).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(seededUuid(makeRng(1))).toBe(a);
    expect(seededUuid(makeRng(2))).not.toBe(a);
  });
});

// ───────────────────────────── records ─────────────────────────────

describe('generateDataset: 2 participants, quick preset', () => {
  it('every session and trial record is schema-valid, simulated, and every stream has equal column lengths', () => {
    expect(ds.sessions).toHaveLength(2);
    for (const s of ds.sessions) {
      expectValid('SessionRecord', s.session);
      expect(s.session.simulated).toBe(true);
      expect(s.session.study_code).toBe('synthetic-v1');
      expect(s.trials.length).toBeGreaterThanOrEqual(24);
      for (const t of s.trials) {
        expectValid('TrialRecord', t);
        expect(t.simulated).toBe(true);
        expect(t.session_id).toBe(s.session.session_id);
        expect(t.participant_id).toBe(s.session.participant_id);
        expectEqualColumnLengths(t.streams.motion as unknown as Record<string, unknown>);
        expectEqualColumnLengths(t.streams.orientation as unknown as Record<string, unknown>);
        expectEqualColumnLengths(t.streams.quaternion as unknown as Record<string, unknown>);
        expect(t.streams.motion.t.length).toBeGreaterThan(60);
      }
    }
  });

  it('attempts and progress are consistent (as sessionFlow.saveAttempt / reconcileProgress compute them)', () => {
    for (const s of ds.sessions) {
      const planned = flattenPlan(s.session.protocol.plan);
      expect(s.session.protocol.n_planned_trials).toBe(planned.length);
      expect(s.session.status).toBe('completed');
      expect(s.session.ended_at).not.toBeNull();
      expect(s.session.progress).toEqual({ next_trial_index: planned.length, n_attempts: s.trials.length, n_accepted: planned.length });
      // the app's own reconciliation finds nothing to repair
      expect(reconcileProgress(s.session, s.trials)).toBe(s.session);
      let k = 0;
      for (const p of planned) {
        // attempts 1..n of every planned trial, in recording order; only the last one is accepted
        let attempt = 1;
        while (k < s.trials.length && s.trials[k].trial_index === p.trial_index) {
          const t = s.trials[k];
          expect(t.attempt).toBe(attempt);
          expect(t.block_index).toBe(p.block_index);
          expect(t.trial_in_block).toBe(p.trial_in_block);
          expect(t.condition).toEqual(p.condition);
          expect(t.outcome.accepted).toBe(t.outcome.status === 'complete');
          const last = k + 1 === s.trials.length || s.trials[k + 1].trial_index !== p.trial_index;
          expect(t.outcome.accepted, `trial ${p.trial_index} attempt ${attempt}`).toBe(last);
          attempt++;
          k++;
        }
        expect(attempt, `planned trial ${p.trial_index} was recorded`).toBeGreaterThan(1);
      }
      expect(k).toBe(s.trials.length);
      expect(new Set(s.trials.map((t) => t.trial_id)).size).toBe(s.trials.length);
      expect(s.attempts.map((a) => a.trial_id)).toEqual(s.trials.map((t) => t.trial_id));
    }
  });

  it('the failure behaviours happen and are handled like the app: redo after no_movement, false starts counted', () => {
    const all = ds.sessions.flatMap((s) => s.trials);
    const attempts = ds.sessions.flatMap((s) => s.attempts);
    const failed = all.filter((t) => t.outcome.status !== 'complete');
    expect(failed.length).toBeGreaterThan(0);
    for (const t of failed) expect(t.outcome.accepted).toBe(false);
    attempts.forEach((a, i) => {
      if (a.behavior === 'no_move') expect(all[i].outcome.status).toBe('no_movement');
      if (a.behavior === 'false_start' && a.truth.false_start_t !== null) expect(all[i].timing.false_starts).toBeGreaterThanOrEqual(1);
    });
    expect(attempts.some((a) => a.behavior === 'false_start')).toBe(true);
    expect(all.some((t) => t.attempt > 1)).toBe(true);
  });

  it('timeline: ids, times and perf clock increase through the session; t0_epoch = time_origin + t0_perf', () => {
    for (const s of ds.sessions) {
      const start = Date.parse(s.session.started_at);
      const end = Date.parse(s.session.ended_at!);
      expect(Date.parse(s.session.consent.accepted_at)).toBeLessThan(start);
      expect(Date.parse(s.session.participant.created_at)).toBeLessThan(start);
      expect(Date.parse(s.session.calibration!.performed_at)).toBeLessThan(start);
      let prevStop = -Infinity;
      let prevCreated = start;
      for (const t of s.trials) {
        expect(t.timing.t0_perf_ms).toBeGreaterThan(prevStop);
        prevStop = t.timing.t0_perf_ms + t.timing.stop_ms;
        const created = Date.parse(t.created_at);
        expect(created).toBeGreaterThanOrEqual(prevCreated);
        prevCreated = created;
        expect(t.timing.t0_epoch_ms - t.timing.t0_perf_ms - s.session.device.time_origin_ms).toBeCloseTo(0, 1);
      }
      expect(end).toBeGreaterThanOrEqual(prevCreated);
    }
  });

  it('profiles: EHI answers fit the handedness and are scored with util.scoreEhi', () => {
    for (const s of ds.sessions) {
      const p = s.session.participant;
      expect(p.nickname).toBeNull();
      expect(p.handedness_self).toBe(s.design.dominant);
      const { lq, cls } = scoreEhi(p.ehi_sf!);
      expect(p.ehi_lq).toBe(lq);
      expect(p.ehi_class).toBe(cls);
      expect(Math.sign(lq)).toBe(s.design.dominant === 'left' ? -1 : 1);
      expect(p.sex).toBe(s.design.sex);
      expect(p.height_cm).toBe(s.sim.participant.height_cm);
    }
    expect(ds.sessions.filter((s) => s.design.dominant === 'left')).toHaveLength(Math.round(2 * 0.3));
  });

  it('platforms: iOS-like (webkit, flat accg_z < 0, sign −1, full precision) and Android-like (blink, +1, 0.1 grid)', () => {
    const ios = ds.sessions.find((s) => s.design.platform === 'ios')!;
    const android = ds.sessions.find((s) => s.design.platform === 'android')!;
    expect(ios.session.device.platform).toBe('ios');
    expect(ios.session.device.engine).toBe('webkit');
    expect(ios.session.capabilities.generic_relative_orientation).toBe(false);
    expect(android.session.device.platform).toBe('android');
    expect(android.session.device.engine).toBe('blink');
    for (const [s, sign] of [
      [ios, -1],
      [android, 1],
    ] as const) {
      const cal = s.session.calibration!;
      expect(cal.flat!.still).toBe(true);
      expect(Math.sign(cal.flat!.accg_mean[2])).toBe(sign);
      expect(Math.abs(cal.flat!.accg_mean[2])).toBeGreaterThan(9.5);
      expect(cal.gravity_sign).toBe(sign);
      expect(cal.upright_check).toBe('ok');
      expect(cal.axis_check).toBeNull();
      expect(s.session.capabilities.measured!.motion_hz).toBeGreaterThan(50);
    }
    // iOS: no quaternion stream; stored motion values are NOT on Chrome's 0.1 grid
    const iosValues = ios.trials.slice(0, 5).flatMap((t) => streamValues(t, ['acc_x', 'accg_y', 'rr_beta']));
    expect(ios.trials.every((t) => t.streams.quaternion === null)).toBe(true);
    expect(nearRound(iosValues, 0.1, 1e-6)).toBeLessThan(0.1);
    // Android: every stored acc/accg/rr value is a multiple of 0.1 (after the app's 3–4 decimal rounding)
    const andValues = android.trials.flatMap((t) => streamValues(t, ['acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma']));
    expect(nearRound(andValues, 0.1, 1e-9)).toBe(1);
    expect(android.trials.every((t) => t.streams.quaternion !== null)).toBe(true);
  });

  it('the foreperiod rng of an attempt is the app\'s makeRng(trialSeed(session seed, trial_index, attempt))', () => {
    let checked = 0;
    let matched = 0;
    for (const s of ds.sessions) {
      const p = s.session.settings.segmentation_params;
      for (const t of s.trials) {
        if (t.timing.false_starts > 0 || t.timing.foreperiod_ms === null) continue;
        const r = makeRng(trialSeed(s.session.protocol.seed, t.trial_index, t.attempt))();
        const fp = Math.round((p.foreperiod_min_ms + r * (p.foreperiod_max_ms - p.foreperiod_min_ms)) * 100) / 100;
        checked++;
        if (Math.abs(fp - t.timing.foreperiod_ms) < 0.006) matched++;
      }
    }
    expect(checked).toBeGreaterThan(30);
    expect(matched / checked).toBeGreaterThan(0.9); // a 'still_lost' (rare) redraws the foreperiod
  });

  it('is deterministic: the same seed gives identical JSON, another seed does not', async () => {
    const again = await generateDataset(SMALL);
    const json = (d: SyntheticDataset) => JSON.stringify(d.sessions.map((s) => [s.session, s.trials]));
    expect(json(again)).toBe(json(ds));
    const other = await generateDataset({ ...SMALL, participants: 1, seed: 2027 });
    expect(other.sessions[0].session.session_id).not.toBe(ds.sessions[0].session.session_id);
    expect(summarizeSession(ds.sessions[0]).n_attempts).toBe(ds.sessions[0].trials.length);
  });
});

describe('generateDataset: options', () => {
  it('custom preset: the stored spec is the normalised one and describes the plan; iosGravitySign +1 is honoured', async () => {
    const custom = {
      orientations: ['upright_away', 'upright_away', 'flat_up'],
      speeds: [],
      start_positions: ['center'],
      reps: 1,
      odd_variants: [],
      include_free: false,
    } as unknown as Parameters<typeof normalizeCustomSpec>[0];
    const d = await generateDataset({ participants: 1, preset: 'custom', custom, seed: 5, failureRate: 0, iosGravitySign: 1 });
    const s = d.sessions[0];
    expectValid('SessionRecord', s.session);
    expect(s.session.protocol.custom).toEqual(normalizeCustomSpec(custom));
    expect(s.session.protocol.n_planned_trials).toBe(2 * 2 * 3); // 2 orientations × 2 hands × 3 speeds
    expect(s.trials).toHaveLength(12);
    expect(s.design.platform).toBe('ios');
    expect(s.session.calibration!.gravity_sign).toBe(1);
    expect(s.session.calibration!.flat!.accg_mean[2]).toBeGreaterThan(9.5);
    for (const t of s.trials) expectValid('TrialRecord', t);
  });
});
