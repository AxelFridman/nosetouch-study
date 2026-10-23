import { beforeEach, describe, expect, it, vi } from 'vitest';

// Collaborators implemented by other agents are replaced by simple fakes (tests/helpers/sessionFakes.ts).
vi.mock('../src/core/protocol', async () => (await import('./helpers/sessionFakes')).protocolModule);
vi.mock('../src/core/util', async () => (await import('./helpers/sessionFakes')).utilModule);
vi.mock('../src/core/config', async () => (await import('./helpers/sessionFakes')).configModule);
vi.mock('../src/core/sensors', async () => (await import('./helpers/sessionFakes')).sensorsModule);
vi.mock('../src/app/services', async () => (await import('./helpers/sessionFakes')).servicesModule);

import { DEFAULT_SEGMENTATION_PARAMS, DEFAULT_STUDY_CONFIG } from '../src/core/constants';
import { FLAT_SIGN_CONTRADICTS_ENGINE } from '../src/core/calibration';
import type { CustomProtocolSpec, SensorSource, SessionRecord } from '../src/core/types';
import { createTapGuard, initialState, TAP_GUARD_MS, type ScreenId } from '../src/app/state';
import { backTargetOf, getState, installHistory, navigate, setBackHandler, setState } from '../src/app/store';
import { persistAttempt, saveForReview } from '../src/app/screens/SessionScreen';
import {
  applyDecision,
  asFailedAttempt,
  asUnreviewed,
  blockOf,
  blockPosition,
  buildTrialIds,
  buildTrialOptions,
  calibrationForDetection,
  createSession,
  finishSession,
  nextAttemptNumber,
  nextPlannedTrial,
  plannedTrials,
  reconcileProgress,
  resumeSession,
  REVIEW_INTERRUPTED_NOTE,
  saveAttempt,
  sessionMinutes,
  stepAfterAccept,
  trialSeed,
  type NewSessionOptions,
} from '../src/app/sessionFlow';
import {
  FAKE_DEFAULT_CUSTOM_SPEC,
  FAKE_RANDOM_SEED,
  installFakeServices,
  makeCapabilities,
  makeDevice,
  makeProfile,
  makeTrialRecord,
  type FakeServices,
} from './helpers/sessionFakes';
import { expectValid } from './helpers/schema';

/** The e2e scenario: upright_toward × natural × natural × 1 rep → one block per hand, one trial each. */
const TWO_TRIALS: CustomProtocolSpec = {
  orientations: ['upright_toward'],
  speeds: ['natural'],
  start_positions: ['natural'],
  reps: 1,
  odd_variants: [],
  include_free: false,
};
const CUSTOM: NewSessionOptions = { preset: 'custom', posture: 'standing', custom: TWO_TRIALS, seed: 7 };
const CONSENT_AT = '2026-10-23T10:05:00.000Z';

let services: FakeServices;

beforeEach(() => {
  services = installFakeServices();
  const base = initialState();
  setState({
    ...base,
    config: { ...DEFAULT_STUDY_CONFIG },
    settings: { ...base.settings, simulate: true, language: 'en' },
    participant: makeProfile(),
    device: makeDevice(),
    capabilities: makeCapabilities(),
    calibration: null,
    consentAcceptedAt: CONSENT_AT,
    session: null,
  });
});

function fakeSource(kind: SensorSource['kind']): SensorSource {
  return {
    kind,
    clock: { now: () => 0, epochAt: (t) => t },
    start: async () => undefined,
    stop: () => undefined,
  };
}

describe('createSession', () => {
  it('builds, snapshots, saves and enqueues a schema-valid session', async () => {
    const s = await createSession(CUSTOM);
    expectValid('SessionRecord', s);
    expect(s.status).toBe('in_progress');
    expect(s.ended_at).toBeNull();
    expect(s.progress).toEqual({ next_trial_index: 0, n_attempts: 0, n_accepted: 0 });
    expect(s.participant_id).toBe('P-ABC234');
    expect(s.study_code).toBe(DEFAULT_STUDY_CONFIG.study_code);
    expect(s.simulated).toBe(true);
    expect(s.consent).toEqual({ version: DEFAULT_STUDY_CONFIG.consent_version, accepted_at: CONSENT_AT });
    expect(s.settings.segmentation_params).toEqual(DEFAULT_SEGMENTATION_PARAMS);
    expect(s.settings.segmentation_params).not.toBe(DEFAULT_SEGMENTATION_PARAMS);
    expect(s.timezone_offset_min).toBe(-new Date().getTimezoneOffset() || 0);
    expect(s.protocol.seed).toBe(7);
    expect(s.protocol.first_hand).toBe('left'); // fake: odd seed → left
    expect(s.protocol.n_planned_trials).toBe(2);
    expect(s.protocol.plan).toHaveLength(2);
    expect(s.protocol.custom).toEqual(TWO_TRIALS);

    expect(getState().session).toBe(s);
    expect(services.store.sessions.get(s.session_id)).toEqual(s);
    expect(services.store.outbox.get(`session:${s.session_id}`)?.status).toBe('pending');
  });

  it('redacts the nickname in the snapshot but not in the app state', async () => {
    const s = await createSession(CUSTOM);
    expect(s.participant.nickname).toBeNull();
    expect(getState().participant?.nickname).toBe('Dana');
  });

  it('keeps its own deep copy of the custom spec', async () => {
    const spec: CustomProtocolSpec = JSON.parse(JSON.stringify(TWO_TRIALS));
    const s = await createSession({ ...CUSTOM, custom: spec });
    spec.orientations.push('upright_away');
    spec.reps = 3;
    expect(s.protocol.custom).toEqual(TWO_TRIALS);
  });

  it('uses DEFAULT_CUSTOM_SPEC for a custom preset without a spec, and no spec for other presets', async () => {
    const custom = await createSession({ preset: 'custom', posture: 'sitting', custom: null, seed: 2 });
    expect(custom.protocol.custom).toEqual(FAKE_DEFAULT_CUSTOM_SPEC);
    expect(custom.protocol.posture).toBe('sitting');
    const quick = await createSession({ preset: 'quick', posture: 'standing', custom: TWO_TRIALS, seed: 2 });
    expect(quick.protocol.custom).toBeNull();
    expect(quick.protocol.n_planned_trials).toBe(plannedTrials(quick).length);
  });

  it('draws a fresh seed when none is given and defaults consent time to now', async () => {
    setState({ consentAcceptedAt: null });
    const before = Date.now();
    const s = await createSession({ ...CUSTOM, seed: undefined });
    expect(s.protocol.seed).toBe(FAKE_RANDOM_SEED);
    expect(Date.parse(s.consent.accepted_at)).toBeGreaterThanOrEqual(before - 1);
    expect(s.consent.accepted_at).toBe(s.started_at);
  });

  it('refuses to start without a participant profile', async () => {
    setState({ participant: null });
    await expect(createSession(CUSTOM)).rejects.toThrow(/participant/);
    expect(services.store.sessions.size).toBe(0);
  });
});

describe('plan queries', () => {
  it('nextPlannedTrial follows progress.next_trial_index and ends with null', async () => {
    const s = await createSession(CUSTOM);
    const trials = plannedTrials(s);
    expect(nextPlannedTrial(s)).toEqual(trials[0]);
    const s1: SessionRecord = { ...s, progress: { ...s.progress, next_trial_index: 1 } };
    expect(nextPlannedTrial(s1)).toEqual(trials[1]);
    expect(nextPlannedTrial({ ...s, progress: { ...s.progress, next_trial_index: 2 } })).toBeNull();
  });

  it('blockOf / blockPosition / stepAfterAccept', async () => {
    const s = await createSession({ ...CUSTOM, custom: { ...TWO_TRIALS, speeds: ['natural', 'fast'] } });
    // 2 blocks × 2 trials
    expect(blockOf(s, 1)?.trials).toHaveLength(2);
    expect(blockOf(s, 9)).toBeNull();
    expect(blockPosition(s, 1)).toEqual({ position: 2, count: 2 });
    const at = (i: number): SessionRecord => ({ ...s, progress: { ...s.progress, next_trial_index: i } });
    expect(stepAfterAccept(at(1), 0)).toBe('trial');
    expect(stepAfterAccept(at(2), 0)).toBe('block_break');
    expect(stepAfterAccept(at(4), 1)).toBe('finished');
  });

  it('sessionMinutes measures start → end', async () => {
    const s = await createSession(CUSTOM);
    const done: SessionRecord = { ...s, started_at: '2026-10-23T10:00:00.000Z', ended_at: '2026-10-23T10:06:30.000Z' };
    expect(sessionMinutes(done)).toBe(6.5);
    expect(sessionMinutes({ ...done, started_at: 'garbage' })).toBeNull();
  });
});

describe('saveAttempt', () => {
  it('accepted attempt: counts it, advances, saves, enqueues both records and kicks an upload', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const t0 = makeTrialRecord(s0, p0, 1);
    expectValid('TrialRecord', t0);
    const s1 = await saveAttempt(s0, t0);
    expectValid('SessionRecord', s1);
    expect(s1.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(Date.parse(s1.updated_at)).toBeGreaterThanOrEqual(Date.parse(s0.updated_at));
    expect(getState().session).toBe(s1);
    expect(services.store.trials.get(t0.trial_id)).toEqual(t0);
    expect(services.store.sessions.get(s0.session_id)?.progress).toEqual(s1.progress);
    expect(services.store.outbox.get(`trial:${t0.trial_id}`)?.status).toBe('pending');
    expect(services.store.outbox.get(`session:${s0.session_id}`)?.status).toBe('pending');
    expect(services.uploader.flush).toHaveBeenCalled();
  });

  it('not accepted (redo / failed): counts the attempt but stays on the same trial', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, p0, 1, { accepted: false }));
    expect(s1.progress).toEqual({ next_trial_index: 0, n_attempts: 1, n_accepted: 0 });
    const s2 = await saveAttempt(s1, makeTrialRecord(s0, p0, 2, { status: 'no_movement' }));
    expect(s2.progress).toEqual({ next_trial_index: 0, n_attempts: 2, n_accepted: 0 });
    expect(nextPlannedTrial(s2)?.trial_index).toBe(0);
  });

  it('re-saving the same attempt never double-counts; acceptance can still change', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const t = makeTrialRecord(s0, p0, 1);
    const protective = asUnreviewed(t);
    const s1 = await saveAttempt(s0, protective);
    expect(s1.progress).toEqual({ next_trial_index: 0, n_attempts: 1, n_accepted: 0 });
    const s2 = await saveAttempt(s1, applyDecision(t, { kind: 'accept' }));
    expect(s2.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    const s3 = await saveAttempt(s2, applyDecision(t, { kind: 'accept' }));
    expect(s3.progress).toEqual(s2.progress);
    expect(services.store.trials.get(t.trial_id)?.outcome.note).toBeNull();
  });

  it('a stale session copy cannot roll progress back', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    await saveAttempt(s0, makeTrialRecord(s0, p0, 1));
    const s2 = await saveAttempt(s0, makeTrialRecord(s0, p1, 1)); // s0 is stale here
    expect(s2.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
  });

  it('concurrent saves are serialized', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    await Promise.all([saveAttempt(s0, makeTrialRecord(s0, p0, 1)), saveAttempt(s0, makeTrialRecord(s0, p1, 1))]);
    expect(getState().session?.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
    expect(services.store.sessions.get(s0.session_id)?.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
  });

  it('never moves past the end of the plan', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    let s = await saveAttempt(s0, makeTrialRecord(s0, p0, 1));
    s = await saveAttempt(s, makeTrialRecord(s0, p1, 1));
    s = await saveAttempt(s, makeTrialRecord(s0, p1, 2)); // an extra accepted attempt
    expect(s.progress).toEqual({ next_trial_index: 2, n_attempts: 3, n_accepted: 3 });
    expect(nextPlannedTrial(s)).toBeNull();
  });

  it('rejects a trial of another session', async () => {
    const s0 = await createSession(CUSTOM);
    const t = { ...makeTrialRecord(s0, plannedTrials(s0)[0], 1), session_id: 'another-session' };
    await expect(saveAttempt(s0, t)).rejects.toThrow(/another session/);
    expect(services.store.trials.size).toBe(0);
  });
});

describe('nextAttemptNumber', () => {
  it('is 1 + the stored attempts of that planned trial', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    expect(await nextAttemptNumber(s0, 0)).toBe(1);
    let s = await saveAttempt(s0, makeTrialRecord(s0, p0, 1, { accepted: false }));
    s = await saveAttempt(s, makeTrialRecord(s0, p0, 2, { status: 'interrupted' }));
    expect(await nextAttemptNumber(s, 0)).toBe(3);
    expect(await nextAttemptNumber(s, p1.trial_index)).toBe(1);
  });

  it('sees attempts stored by other means, too', async () => {
    const s0 = await createSession(CUSTOM);
    await services.store.saveTrial(makeTrialRecord(s0, plannedTrials(s0)[0], 1));
    expect(await nextAttemptNumber(s0, 0)).toBe(2);
  });
});

describe('finishSession', () => {
  it('stamps the end, saves, enqueues and is final', async () => {
    const s0 = await createSession(CUSTOM);
    const done = await finishSession(s0, 'completed');
    expectValid('SessionRecord', done);
    expect(done.status).toBe('completed');
    expect(done.ended_at).not.toBeNull();
    expect(done.updated_at).toBe(done.ended_at);
    expect(getState().session).toBe(done);
    expect(services.store.sessions.get(s0.session_id)?.status).toBe('completed');
    expect(services.store.outbox.get(`session:${s0.session_id}`)?.status).toBe('pending');
    const again = await finishSession(s0, 'abandoned');
    expect(again.status).toBe('completed');
    expect(again.ended_at).toBe(done.ended_at);
  });

  it('abandoned keeps the recorded progress', async () => {
    const s0 = await createSession(CUSTOM);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, plannedTrials(s0)[0], 1));
    const ended = await finishSession(s0, 'abandoned');
    expect(ended.status).toBe('abandoned');
    expect(ended.progress).toEqual(s1.progress);
  });
});

describe('resumeSession', () => {
  it('loads an in-progress session into the state', async () => {
    const s0 = await createSession(CUSTOM);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, plannedTrials(s0)[0], 1));
    setState({ session: null });
    const resumed = await resumeSession(s0.session_id);
    expect(resumed).toEqual(s1);
    expect(getState().session).toEqual(s1);
    expect(nextPlannedTrial(resumed!)?.trial_index).toBe(1);
  });

  it('returns null for unknown and finished sessions', async () => {
    expect(await resumeSession('nope')).toBeNull();
    const s0 = await createSession(CUSTOM);
    await finishSession(s0, 'completed');
    setState({ session: null });
    expect(await resumeSession(s0.session_id)).toBeNull();
    expect(getState().session).toBeNull();
  });

  it('repairs progress when the app died between saving a trial and saving the session', async () => {
    const s0 = await createSession(CUSTOM);
    await services.store.saveTrial(makeTrialRecord(s0, plannedTrials(s0)[0], 1)); // session never updated
    services.store.outbox.clear();
    setState({ session: null });
    const resumed = await resumeSession(s0.session_id);
    expect(resumed?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(services.store.sessions.get(s0.session_id)?.progress).toEqual(resumed?.progress);
    expect(services.store.outbox.get(`session:${s0.session_id}`)?.status).toBe('pending');
  });
});

describe('reconcileProgress', () => {
  it('returns the same object when nothing changes', async () => {
    const s0 = await createSession(CUSTOM);
    expect(reconcileProgress(s0, [])).toBe(s0);
  });

  it('only moves next_trial_index forward and ignores other sessions', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const ahead: SessionRecord = { ...s0, progress: { next_trial_index: 1, n_attempts: 0, n_accepted: 0 } };
    const foreign = { ...makeTrialRecord(s0, p0, 1), session_id: 'x' };
    const r = reconcileProgress(ahead, [makeTrialRecord(s0, p0, 1, { accepted: false }), foreign]);
    expect(r.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 0 });
  });
});

describe('trial set-up helpers', () => {
  it('trialSeed is a deterministic uint32 that differs per trial and attempt', () => {
    const a = trialSeed(123, 4, 1);
    expect(trialSeed(123, 4, 1)).toBe(a);
    expect(Number.isInteger(a) && a >= 0 && a <= 0xffffffff).toBe(true);
    const seeds = new Set<number>();
    for (let i = 0; i < 50; i++) for (let k = 1; k <= 4; k++) seeds.add(trialSeed(123, i, k));
    expect(seeds.size).toBe(200);
    expect(trialSeed(124, 4, 1)).not.toBe(a);
  });

  it('buildTrialIds copies the planned position and the attempt', async () => {
    const s = await createSession(CUSTOM);
    const p1 = plannedTrials(s)[1];
    expect(buildTrialIds(s, p1, 3, 'trial-x')).toEqual({
      trial_id: 'trial-x',
      session_id: s.session_id,
      participant_id: s.participant_id,
      study_code: s.study_code,
      block_index: p1.block_index,
      trial_index: p1.trial_index,
      trial_in_block: p1.trial_in_block,
      attempt: 3,
    });
    expect(buildTrialIds(s, p1, 1).trial_id).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
  });

  it('buildTrialOptions resolves params, seeds the rng per attempt and labels simulated data by source', async () => {
    const s = await createSession({ ...CUSTOM, custom: { ...TWO_TRIALS, speeds: ['fast'] } });
    const p0 = plannedTrials(s)[0];
    const cues = { play: vi.fn() };
    const o = buildTrialOptions(s, p0, 2, { source: fakeSource('simulated'), cues, trialId: 'tid' });
    expect(o.params.onset_sustain_ms).toBe(40); // fast override from resolveParams
    expect(o.condition).toEqual(p0.condition);
    expect(o.calibration).toBe(s.calibration);
    expect(o.ids.attempt).toBe(2);
    expect(o.ids.trial_id).toBe('tid');
    expect(o.simulated).toBe(true);
    expect(o.cues).toBe(cues);
    const again = buildTrialOptions(s, p0, 2, { source: fakeSource('simulated'), cues });
    expect([o.rng(), o.rng(), o.rng()]).toEqual([again.rng(), again.rng(), again.rng()]);
    const other = buildTrialOptions(s, p0, 3, { source: fakeSource('browser'), cues });
    expect(other.simulated).toBe(false);
    expect(other.rng()).not.toBe(buildTrialOptions(s, p0, 2, { source: fakeSource('browser'), cues }).rng());
  });

  it('orientation detection does not trust a flat sign that contradicts the engine (screen-down calibration)', async () => {
    const s0 = await createSession({ ...CUSTOM, custom: TWO_TRIALS });
    const contradicted: SessionRecord = {
      ...s0,
      device: { ...s0.device, engine: 'blink' },
      calibration: {
        performed_at: s0.started_at,
        flat: null,
        upright: null,
        gravity_sign: -1, // a screen-down flat step on Android
        gravity_magnitude: 9.8,
        upright_check: 'inverted',
        gyro_bias_dps: null,
        notes: [FLAT_SIGN_CONTRADICTS_ENGINE],
        axis_check: null,
      },
    };
    const p0 = plannedTrials(contradicted)[0];
    const o = buildTrialOptions(contradicted, p0, 1, { source: fakeSource('browser'), cues: { play: vi.fn() } });
    expect(o.calibration?.gravity_sign).toBe(1); // blink's known sign
    expect(contradicted.calibration?.gravity_sign).toBe(-1); // the stored calibration is unchanged
    const ios: SessionRecord = { ...contradicted, device: { ...contradicted.device, engine: 'webkit' } };
    expect(calibrationForDetection(ios)?.gravity_sign).toBe(-1);
    const unknown: SessionRecord = { ...contradicted, device: { ...contradicted.device, engine: 'unknown' } };
    expect(calibrationForDetection(unknown)?.gravity_sign).toBeNull();
  });

  it('orientation detection uses the engine sign when calibration was skipped or measured no sign', async () => {
    const s0 = await createSession({ ...CUSTOM, custom: TWO_TRIALS });
    const skipped: SessionRecord = { ...s0, device: { ...s0.device, engine: 'webkit' }, calibration: null };
    const p0 = plannedTrials(skipped)[0];
    const o = buildTrialOptions(skipped, p0, 1, { source: fakeSource('browser'), cues: { play: vi.fn() } });
    expect(o.calibration?.gravity_sign).toBe(-1); // WebKit's known sign, not the W3C default
    expect(skipped.calibration).toBeNull(); // nothing is stored
    const android: SessionRecord = { ...skipped, device: { ...skipped.device, engine: 'blink' } };
    expect(calibrationForDetection(android)?.gravity_sign).toBe(1);
    const unknown: SessionRecord = { ...skipped, device: { ...skipped.device, engine: 'unknown' } };
    expect(calibrationForDetection(unknown)).toBeNull();

    const noSign: SessionRecord = {
      ...skipped,
      calibration: {
        performed_at: s0.started_at,
        flat: null,
        upright: null,
        gravity_sign: null, // the flat step failed (moving, not flat)
        gravity_magnitude: null,
        upright_check: null,
        gyro_bias_dps: null,
        notes: ['flat: moving'],
        axis_check: null,
      },
    };
    expect(calibrationForDetection(noSign)?.gravity_sign).toBe(-1);
    expect(calibrationForDetection(noSign)?.notes).toEqual(['flat: moving']);
    expect(noSign.calibration?.gravity_sign).toBeNull(); // the stored calibration is unchanged

    const measured: SessionRecord = { ...noSign, calibration: { ...noSign.calibration!, gravity_sign: -1, notes: [] } };
    expect(calibrationForDetection(measured)).toBe(measured.calibration); // a trusted measurement passes through
  });

  it('applyDecision / asUnreviewed / asFailedAttempt', async () => {
    const s = await createSession(CUSTOM);
    const t = makeTrialRecord(s, plannedTrials(s)[0], 1);
    expect(applyDecision(t, { kind: 'accept' }).outcome).toEqual({ ...t.outcome, accepted: true, participant_flag: null });
    expect(applyDecision(t, { kind: 'redo' }).outcome.accepted).toBe(false);
    const flagged = applyDecision(t, { kind: 'flag', flag: 'wrong_hand' });
    expect(flagged.outcome).toMatchObject({ accepted: false, participant_flag: 'wrong_hand' });
    expectValid('TrialRecord', flagged);
    const failed = makeTrialRecord(s, plannedTrials(s)[0], 1, { status: 'too_long', accepted: false });
    expect(applyDecision(failed, { kind: 'accept' }).outcome.accepted).toBe(false);

    const unreviewed = asUnreviewed(t);
    expect(unreviewed.outcome).toMatchObject({ accepted: false, note: REVIEW_INTERRUPTED_NOTE });
    expect(t.outcome.accepted).toBe(true); // input untouched
    expect(applyDecision(unreviewed, { kind: 'accept' }).outcome.note).toBeNull();
    expectValid('TrialRecord', unreviewed);

    const f = asFailedAttempt({ ...t, outcome: { ...t.outcome, status: 'interrupted' } }, 'ui_abort:sensor_stall');
    expect(f.outcome.accepted).toBe(false);
    expect(f.quality.warnings).toEqual(['ui_abort:sensor_stall']);
    expect(t.quality.warnings).toEqual([]);
    expect(asFailedAttempt(t, null).quality.warnings).toEqual([]);
  });
});

describe('a complete attempt is saved before its review opens (SessionScreen)', () => {
  it('is stored at once as not accepted; the decision re-saves the same trial and counts it once', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const rec = makeTrialRecord(s0, p0, 1); // what the controller delivers: complete
    const settled = new Set<string>();
    const s1 = await saveForReview(rec, settled);
    expect(settled.has(rec.trial_id)).toBe(true); // nothing left for the protective save on pagehide
    expect(s1?.progress).toEqual({ next_trial_index: 0, n_attempts: 1, n_accepted: 0 });
    expect(services.store.trials.get(rec.trial_id)?.outcome).toMatchObject({ accepted: false, note: REVIEW_INTERRUPTED_NOTE });
    expect(services.store.outbox.get(`trial:${rec.trial_id}`)?.status).toBe('pending');
    expectValid('TrialRecord', services.store.trials.get(rec.trial_id));

    const s2 = await persistAttempt(applyDecision(rec, { kind: 'accept' }));
    expect(s2?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(services.store.trials.size).toBe(1);
    expect(services.store.trials.get(rec.trial_id)?.outcome).toMatchObject({ accepted: true, note: null });
    expect(services.store.sessions.get(s0.session_id)?.progress).toEqual(s2?.progress);
  });

  it('Redo and Problem keep one attempt and the same planned trial; the next attempt is number 2', async () => {
    for (const decision of [{ kind: 'redo' }, { kind: 'flag', flag: 'wrong_hand' }] as const) {
      const s0 = await createSession(CUSTOM);
      const [p0] = plannedTrials(s0);
      const rec = makeTrialRecord(s0, p0, 1);
      await saveForReview(rec, new Set());
      const s2 = await persistAttempt(applyDecision(rec, decision));
      expect(s2?.progress).toEqual({ next_trial_index: 0, n_attempts: 1, n_accepted: 0 });
      expect(services.store.trials.get(rec.trial_id)?.outcome.note).toBeNull();
      expect(await nextAttemptNumber(s2!, 0)).toBe(2);
    }
  });

  it('a decision taken while the first save is still running is applied after it', async () => {
    const s0 = await createSession(CUSTOM);
    const rec = makeTrialRecord(s0, plannedTrials(s0)[0], 1);
    await Promise.all([saveForReview(rec, new Set()), persistAttempt(applyDecision(rec, { kind: 'accept' }))]);
    expect(getState().session?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(services.store.trials.get(rec.trial_id)?.outcome.accepted).toBe(true);
  });

  it('when the first save failed, the decision still stores the attempt (counted once)', async () => {
    const s0 = await createSession(CUSTOM);
    const rec = makeTrialRecord(s0, plannedTrials(s0)[0], 1);
    const realSave = services.store.saveTrial;
    services.store.saveTrial = vi.fn(async () => {
      throw new Error('QuotaExceededError');
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      expect(await saveForReview(rec, new Set())).toBeNull();
      expect(getState().toast?.kind).toBe('error');
    } finally {
      services.store.saveTrial = realSave;
      quiet.mockRestore();
    }
    const s2 = await persistAttempt(applyDecision(rec, { kind: 'accept' }));
    expect(s2?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
  });
});

describe('tap guard (double taps on buttons that replace each other)', () => {
  it('ignores taps for TAP_GUARD_MS after what is on screen changed, and only then', () => {
    let now = 1000;
    const guard = createTapGuard(() => now);
    expect(guard.tooSoon()).toBe(false); // nothing shown yet
    guard.shown('intro');
    now += 200; // the second tap of a double tap
    expect(guard.tooSoon()).toBe(true);
    now += TAP_GUARD_MS - 200;
    expect(guard.tooSoon()).toBe(false);
    guard.shown('intro'); // a re-render of the same step does not restart it
    expect(guard.tooSoon()).toBe(false);
    guard.shown('ready');
    now += 1;
    expect(guard.tooSoon()).toBe(true);
    now += TAP_GUARD_MS;
    expect(guard.tooSoon()).toBe(false);
  });
});

/** history as the browser keeps it for one tab: entries, popstate on traversal, leaving before the first. */
class FakeHistory {
  entries: unknown[] = [null];
  index = 0;
  left = false;
  /** traversals that never deliver their popstate (to test that the store does not wait forever) */
  silent = false;
  constructor(private readonly target: EventTarget) {}
  get state(): unknown {
    return this.entries[this.index];
  }
  pushState(s: unknown): void {
    this.entries = [...this.entries.slice(0, this.index + 1), structuredClone(s)];
    this.index++;
  }
  replaceState(s: unknown): void {
    this.entries[this.index] = structuredClone(s);
  }
  back(): void {
    this.go(-1);
  }
  go(delta: number): void {
    const i = this.index + delta;
    if (i < 0) {
      this.left = true; // the page before the app (or the PWA closes)
      return;
    }
    if (i >= this.entries.length) return;
    this.index = i;
    if (this.silent) return;
    const ev = new Event('popstate');
    Object.defineProperty(ev, 'state', { value: this.state });
    setTimeout(() => this.target.dispatchEvent(ev), 0);
  }
}

describe('system back button (history guard)', () => {
  const settle = () => new Promise((r) => setTimeout(r, 5));
  let fake: FakeHistory;
  let uninstall: () => void;
  const g = globalThis as { history?: unknown };

  beforeEach(() => {
    const target = new EventTarget();
    fake = new FakeHistory(target);
    g.history = fake;
    setState({ screen: 'loading', returnTo: 'welcome' });
    uninstall = installHistory(target);
    navigate('welcome'); // what boot does
    return () => {
      uninstall();
      delete g.history;
    };
  });

  const press = async () => {
    fake.back();
    await settle();
  };

  it('a forward move pushes ONE guard entry; later moves replace it', () => {
    expect(fake.entries).toHaveLength(1);
    for (const screen of ['consent', 'profile', 'sensors', 'calibration', 'setup'] as ScreenId[]) {
      navigate(screen);
      expect(fake.entries).toHaveLength(2);
      expect(fake.index).toBe(1);
      expect(fake.state).toMatchObject({ screen });
    }
  });

  it("back walks the screens' ← arrows and leaves the app only from the root", async () => {
    for (const screen of ['consent', 'profile', 'sensors', 'calibration', 'setup'] as ScreenId[]) navigate(screen);
    for (const expected of ['calibration', 'sensors', 'profile', 'welcome'] as ScreenId[]) {
      await press();
      expect(getState().screen).toBe(expected);
      expect(fake.left).toBe(false);
    }
    expect(fake.index).toBe(0); // the root keeps no guard
    await press();
    expect(fake.left).toBe(true);
  });

  it('returning to the root in the app drops the guard, so back there leaves at once', async () => {
    navigate('consent');
    navigate('welcome'); // e.g. "Not now", or Pause in a session
    await settle();
    expect(fake.index).toBe(0);
    expect(getState().screen).toBe('welcome');
    await press();
    expect(fake.left).toBe(true);
  });

  it('the session screen takes the press and keeps a guard, however often back is pressed', async () => {
    const handler = vi.fn(() => true);
    const remove = setBackHandler(handler);
    try {
      navigate('setup');
      navigate('session');
      for (let i = 1; i <= 3; i++) {
        await press();
        expect(handler).toHaveBeenCalledTimes(i);
        expect(getState().screen).toBe('session');
        expect(fake.index).toBe(1);
        expect(fake.left).toBe(false);
      }
    } finally {
      remove();
    }
    remove(); // removing twice is harmless
    await press();
    expect(getState().screen).toBe('welcome'); // no handler: the fallback target
  });

  it('a handler that declines (no session running) lets back go to the fallback target', async () => {
    const remove = setBackHandler(() => false);
    try {
      navigate('session');
      await press();
      expect(getState().screen).toBe('welcome');
    } finally {
      remove();
    }
  });

  it('a handler removed late does not remove a newer one', async () => {
    const first = setBackHandler(() => true);
    const second = vi.fn(() => true);
    const removeSecond = setBackHandler(second); // the next screen registered before the old one unmounted
    try {
      first();
      navigate('session');
      await press();
      expect(second).toHaveBeenCalledOnce();
    } finally {
      removeSecond();
    }
  });

  it('done goes back to the start screen like its Home button; menu screens go where they came from', async () => {
    setState({ session: { session_id: 'finished' } as SessionRecord });
    navigate('done');
    await press();
    expect(getState()).toMatchObject({ screen: 'welcome', session: null });
    navigate('data', { returnTo: 'welcome' });
    await press();
    expect(getState().screen).toBe('welcome');
    expect(backTargetOf({ screen: 'sandbox', returnTo: 'about' })).toBe('about');
    expect(backTargetOf({ screen: 'about', returnTo: 'about' })).toBe('welcome');
    expect(backTargetOf({ screen: 'welcome', returnTo: 'welcome' })).toBeNull();
    expect(backTargetOf({ screen: 'loading', returnTo: 'welcome' })).toBeNull();
  });

  it('forward onto an old guard at the root is undone', async () => {
    navigate('consent');
    navigate('welcome');
    await settle();
    fake.go(1); // the browser's forward button
    await settle();
    expect(fake.index).toBe(0);
    expect(getState().screen).toBe('welcome');
  });

  it('a popstate of its own that never arrives does not swallow a later press', async () => {
    navigate('consent');
    fake.silent = true;
    navigate('welcome'); // drops the guard; the browser never reports it
    fake.silent = false;
    const now = vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 5000);
    try {
      navigate('profile');
      await press();
      expect(getState().screen).toBe('welcome');
    } finally {
      now.mockRestore();
    }
  });

  it("a reload on a guarded screen turns the restored guard into the page's own entry", () => {
    uninstall();
    fake.entries = [null, { nosetouch: 'nosetouch-guard', screen: 'setup' }];
    fake.index = 1;
    uninstall = installHistory(new EventTarget());
    expect(fake.state).toBeNull();
  });
});

describe('a whole two-trial session', () => {
  it('produces only schema-valid records and a valid export bundle', async () => {
    let s = await createSession(CUSTOM);
    let next = nextPlannedTrial(s);
    while (next) {
      const attempt = await nextAttemptNumber(s, next.trial_index);
      const rec = makeTrialRecord(s, next, attempt);
      s = await saveAttempt(s, applyDecision(rec, { kind: 'accept' }));
      next = nextPlannedTrial(s);
    }
    s = await finishSession(s, 'completed');
    expect(s.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
    const bundle = await services.store.exportBundle({ sessionIds: [s.session_id], redact: (p) => p });
    expectValid('ExportBundle', bundle);
    expect(bundle.trials.map((t) => t.condition.hand).sort()).toEqual(['left', 'right']);
  });
});
