/**
 * services.simulatedSourceOptions: the options of the simulated participant behind ?sim=1. A trial that
 * waits for the return of the arm (awaitReturn) must get a simulated participant who starts at the nose
 * (startAtNose), or the trial waits for a return movement that never comes until the no-GO pause.
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_CONDITION, simulatedSourceOptions } from '../src/app/services';
import type { MotionSample, ParticipantProfile, TrialCondition } from '../src/core/types';
import { createSimulatedSource } from '../src/sim/simulatedSource';

const PROFILE: ParticipantProfile = {
  participant_id: 'P-ABC234',
  nickname: 'Dana Levi',
  age_years: 30,
  sex: 'female',
  handedness_self: 'right',
  ehi_sf: null,
  ehi_lq: null,
  ehi_class: null,
  phone_hand: 'left',
  height_cm: 168,
  arm_issue: false,
  phone_model: null,
  phone_case: null,
  created_at: '2026-10-24T08:00:00.000Z',
  updated_at: '2026-10-24T08:00:00.000Z',
};

const CONDITION: TrialCondition = { ...DEFAULT_CONDITION, hand: 'left', speed: 'fast', rep: 1 };

describe('simulatedSourceOptions', () => {
  it('passes startAtNose through (default: the arm is already out)', () => {
    expect(simulatedSourceOptions({ condition: CONDITION, startAtNose: true }, PROFILE, 7).startAtNose).toBe(true);
    expect(simulatedSourceOptions({ condition: CONDITION, startAtNose: false }, PROFILE, 7).startAtNose).toBe(false);
    expect(simulatedSourceOptions({ condition: CONDITION }, PROFILE, 7).startAtNose).toBe(false);
  });

  it('uses the requested condition and behaviour, else the defaults; realtime; the given seed', () => {
    const o = simulatedSourceOptions({ condition: CONDITION, behavior: 'static' }, PROFILE, 42);
    expect(o).toMatchObject({ mode: 'realtime', condition: CONDITION, behavior: 'static', seed: 42 });
    const d = simulatedSourceOptions({}, PROFILE, 1);
    expect(d.condition).toEqual(DEFAULT_CONDITION);
    expect(d.behavior).toBe('normal');
  });

  it('keeps one simulated person per participant (same style on every trial) from the profile', () => {
    const a = simulatedSourceOptions({}, PROFILE, 1);
    const b = simulatedSourceOptions({}, PROFILE, 2);
    expect(a.participant).toEqual(b.participant);
    expect(a.platform).toEqual(b.platform);
    expect(a.participant.phone_hand).toBe('left');
    expect(a.participant.handedness).toBe('right');
    const other = simulatedSourceOptions({}, { ...PROFILE, participant_id: 'P-XYZ789' }, 1);
    expect(other.participant).not.toEqual(a.participant);
    // no profile yet (the sensor check before the profile was saved): still a valid person
    expect(simulatedSourceOptions({}, null, 1).participant.participant_id).toBeTruthy();
  });

  it('startAtNose makes the simulated source bring the phone back out from the nose first', async () => {
    const run = async (startAtNose: boolean) => {
      const opts = simulatedSourceOptions({ condition: CONDITION, startAtNose }, PROFILE, 11);
      const source = createSimulatedSource({ ...opts, mode: 'offline', maxDurationMs: 3000 });
      const samples: MotionSample[] = [];
      await source.start({ onMotion: (s) => samples.push(s), onOrientation: () => undefined, onError: () => undefined });
      return { truth: source.truth, samples };
    };
    const atNose = await run(true);
    const armOut = await run(false);
    expect(atNose.truth.initial_return_t).toEqual(expect.any(Number));
    expect(armOut.truth.initial_return_t ?? null).toBeNull();
    // the return from the nose is a real movement: far more rotation than the still arm's tremor
    const peak = (xs: MotionSample[]) => Math.max(...xs.map((s) => (s.rr ? Math.hypot(s.rr.alpha, s.rr.beta, s.rr.gamma) : 0)));
    expect(peak(atNose.samples)).toBeGreaterThan(45);
    expect(peak(armOut.samples)).toBeLessThan(20);
  });
});
