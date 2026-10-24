import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Collaborators implemented by other agents are replaced by simple fakes (tests/helpers/sessionFakes.ts).
vi.mock('../src/core/protocol', async () => (await import('./helpers/sessionFakes')).protocolModule);
vi.mock('../src/core/util', async () => (await import('./helpers/sessionFakes')).utilModule);
vi.mock('../src/core/config', async () => (await import('./helpers/sessionFakes')).configModule);
vi.mock('../src/core/sensors', async () => (await import('./helpers/sessionFakes')).sensorsModule);
vi.mock('../src/app/services', async () => (await import('./helpers/sessionFakes')).servicesModule);

import { DEFAULT_SEGMENTATION_PARAMS, DEFAULT_STUDY_CONFIG } from '../src/core/constants';
import { buildCalibration, FLAT_SIGN_CONTRADICTS_ENGINE, UPRIGHT_SIGN_CONTRADICTS_ENGINE } from '../src/core/calibration';
import type {
  CalibrationResult,
  CustomProtocolSpec,
  PlannedTrial,
  SensorSource,
  SessionRecord,
  TrialCondition,
  TrialRecord,
} from '../src/core/types';
import { createTapGuard, initialState, TAP_GUARD_MS, type ScreenId } from '../src/app/state';
import { backTargetOf, getState, installHistory, navigate, setBackHandler, setState } from '../src/app/store';
import {
  ATTEMPT_FLOOR,
  ATTEMPT_READ_RETRY_MS,
  ATTEMPT_READ_TRIES,
  createSessionRunner,
  IDLE_MS,
  persistAttempt,
  RETRY_FLASH_MS,
  saveForReview,
  SPEECH_DUCK_TAIL_MS,
  speechMs,
  WAKE_LOCK_NOT_HELD,
  type RecorderStart,
  type SessionRunner,
  type UiAbort,
} from '../src/app/sessionRunner';
import { WRONG_GRIP_AT_STOP } from '../src/core/trialController';
import { RESUME_MAX_AGE_MS } from '../src/core/storage';
import {
  abandonOtherSessions,
  abandonStaleSessions,
  introWording,
  READ_SPECIAL_MS,
  READ_SPEED_MS,
  readTimeMs,
  recentSessionOf,
  RESUME_ON_CONTINUE_MS,
  applyDecision,
  asFailedAttempt,
  asUnreviewed,
  attemptSummary,
  AUTO_ACCEPT_NOTE,
  AUTO_REJECT_ORIENTATION_NOTE,
  autoDecision,
  awaitingHandReport,
  blockChange,
  blockOf,
  blockPosition,
  blockSpeeds,
  buildTrialIds,
  buildTrialOptions,
  calibrationForDetection,
  createSession,
  endStatusOf,
  finishSession,
  HAND_NOT_REPORTED_NOTE,
  HAND_UNSURE_NOTE,
  isChoiceTrial,
  nextAttemptNumber,
  nextPlannedTrial,
  nextStep,
  plannedTrials,
  progressOf,
  reconcileProgress,
  REDO_REQUESTED_NOTE,
  requiredTrialsLeft,
  resumeSession,
  REVIEW_INTERRUPTED_NOTE,
  reviewModeOf,
  revokeAccepted,
  roundMinutes,
  saveAttempt,
  secondsLeft,
  sessionMinutes,
  setPosture,
  stepAfterAccept,
  trialSeed,
  withReportedHand,
  type NewSessionOptions,
} from '../src/app/sessionFlow';
import { en } from '../src/i18n/en';
import {
  FAKE_DEFAULT_CUSTOM_SPEC,
  FAKE_RANDOM_SEED,
  FAKE_STREET,
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
/** the fake street plan (helpers/sessionFakes.ts): odd seed → first hand A = left, B = right */
const STREET: NewSessionOptions = { preset: 'street', posture: 'standing', custom: null, seed: 1 };
const CONSENT_AT = '2026-10-23T10:05:00.000Z';
/** trial timing of a trial whose return of the arm was observed */
const RETURNED = { timing: { return_onset_ms: 2100, return_end_ms: 2600 } };
/** the attempt never reached GO */
const NO_GO = { status: 'aborted' as const, timing: { still_ms: null, go_ms: null } };

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

afterEach(() => {
  vi.useRealTimers();
});

function fakeSource(kind: SensorSource['kind']): SensorSource {
  return {
    kind,
    clock: { now: () => 0, epochAt: (t) => t },
    start: async () => undefined,
    stop: () => undefined,
  };
}

/** A real (not simulated) session on a phone with this browser engine. */
function onEngine(s: SessionRecord, engine: 'webkit' | 'blink' | 'gecko' | 'unknown'): SessionRecord {
  return { ...s, simulated: false, device: { ...s.device, engine } };
}

/**
 * Lets every save already started finish (sessionFlow runs them one at a time), then the callbacks
 * after them, a few rounds deep (a finish queued by a callback, and so on). Timer-free, so it also
 * works under fake timers.
 */
async function settle(): Promise<void> {
  for (let round = 0; round < 3; round++) {
    const cur = getState().session;
    if (cur) await attemptSummary(cur);
    for (let i = 0; i < 20; i++) await Promise.resolve();
  }
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
    expect(services.uploader.flush).toHaveBeenCalled(); // sent at once, not at the next timer tick
  });

  it('redacts the nickname in the snapshot unless the study uploads names', async () => {
    setState({ config: { ...DEFAULT_STUDY_CONFIG, upload_nickname: false } });
    const s = await createSession(CUSTOM);
    expect(s.participant.nickname).toBeNull();
    expect(getState().participant?.nickname).toBe('Dana');
    setState({ config: { ...DEFAULT_STUDY_CONFIG, upload_nickname: true } }); // v2: names are uploaded
    expect((await createSession(CUSTOM)).participant.nickname).toBe('Dana');
  });

  it('keeps its own deep copy of the custom spec', async () => {
    const spec: CustomProtocolSpec = JSON.parse(JSON.stringify(TWO_TRIALS));
    const s = await createSession({ ...CUSTOM, custom: spec });
    spec.orientations.push('top_left_toward');
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

  it('stores an empty implicit calibration (implicit_v1); a guided one from the state is kept', async () => {
    const s = await createSession(STREET);
    expect(s.calibration).toEqual({
      method: 'implicit_v1',
      performed_at: s.started_at,
      flat: null,
      upright: null,
      gravity_sign: null,
      gravity_magnitude: null,
      upright_check: null,
      gyro_bias_dps: null,
      notes: [],
      axis_check: null,
    });
    expectValid('SessionRecord', s);
    const guided = buildCalibration({ performedAt: CONSENT_AT, flat: null, upright: null, axis: null });
    setState({ calibration: guided });
    expect((await createSession(STREET)).calibration).toEqual(guided);
    expect(guided.method).toBe('guided_v1');
  });

  it('records the review mode: hands-free (auto) unless the researcher setting asks for a tap review', async () => {
    const auto = await createSession(STREET);
    expect(auto.settings.review_mode).toBe('auto');
    expect(reviewModeOf(auto)).toBe('auto');
    setState({ settings: { ...getState().settings, review_each_trial: true } });
    const tap = await createSession(STREET);
    expect(tap.settings.review_mode).toBe('tap');
    expect(reviewModeOf(tap)).toBe('tap');
    expectValid('SessionRecord', tap);
    // sessions from before schema 1.2 have no review_mode: they were all reviewed with a tap
    const { review_mode: _gone, ...v1Settings } = tap.settings;
    expect(reviewModeOf({ ...tap, settings: v1Settings })).toBe('tap');
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

describe('street plan helpers (tiers, the choice trial, the optional bonus)', () => {
  it('the choice trial is the one whose hand is reported', async () => {
    const s = await createSession(STREET);
    const trials = plannedTrials(s);
    expect(trials).toHaveLength(FAKE_STREET.trials);
    expect(isChoiceTrial(trials[0].condition)).toBe(true);
    expect(trials.slice(1).every((t) => !isChoiceTrial(t.condition))).toBe(true);
  });

  it('blockChange: what the intro must say (hand and grip changes, the first block after the basics)', async () => {
    const s = await createSession(STREET);
    expect(blockChange(s, 0)).toEqual({ first: true, handChanged: true, gripChanged: true, afterBasics: false });
    // after the choice block the hand and grip are always announced (the participant chose them)
    expect(blockChange(s, 1)).toEqual({ first: false, handChanged: true, gripChanged: true, afterBasics: false });
    expect(blockChange(s, 2)).toEqual({ first: false, handChanged: true, gripChanged: false, afterBasics: false });
    // same hand, new grip; the first block after tier 1
    expect(blockChange(s, 3)).toEqual({ first: false, handChanged: false, gripChanged: true, afterBasics: true });
    expect(blockChange(s, 4)).toMatchObject({ handChanged: true, gripChanged: true, afterBasics: false });
    expect(blockSpeeds(blockOf(s, 2)!)).toEqual(['natural', 'slow']);
  });

  it('nextStep: the chain goes on inside a block; another block needs its intro; the bonus needs the checkpoint', async () => {
    const s = await createSession(STREET);
    expect(nextStep(s, 2, 1, false)).toBe('trial'); // trial 2 is in block 1
    expect(nextStep(s, 3, 1, false)).toBe('intro'); // trial 3 opens block 2
    expect(nextStep(s, 2, null, false)).toBe('intro'); // after a pause or a reload: the intro first
    expect(nextStep(s, FAKE_STREET.firstOptionalTrial, 4, false)).toBe('checkpoint');
    expect(nextStep(s, FAKE_STREET.firstOptionalTrial, 4, true)).toBe('intro');
    expect(nextStep(s, FAKE_STREET.trials, 6, true)).toBe('finished');
  });

  it('progress and the end status count the non-optional blocks (the bonus only after opting in)', async () => {
    const s = await createSession(STREET);
    expect(progressOf(s, 0, false)).toEqual({ done: 0, total: FAKE_STREET.required });
    expect(progressOf(s, 3, false)).toEqual({ done: 3, total: FAKE_STREET.required });
    expect(progressOf(s, 8, true)).toEqual({ done: 8, total: FAKE_STREET.trials });
    expect(requiredTrialsLeft(s)).toBe(FAKE_STREET.required);
    const at = (i: number): SessionRecord => ({ ...s, progress: { ...s.progress, next_trial_index: i } });
    expect(requiredTrialsLeft(at(5))).toBe(2);
    expect(endStatusOf(at(FAKE_STREET.required - 1))).toBe('ended_early');
    expect(endStatusOf(at(FAKE_STREET.firstOptionalTrial))).toBe('completed');
    expect(endStatusOf(at(FAKE_STREET.trials))).toBe('completed');
  });

  it('secondsLeft follows the estimateMinutes model (6.5 / 5.5 / 10 s per trial, +2 s to read, 6 s per intro)', async () => {
    const s = await createSession(STREET);
    // required trials 1…6: natural, fast | natural, slow | natural | natural+tilted_touch (read); 3 more intros
    const required = 6.5 + 5.5 + 6.5 + 10 + 6.5 + (6.5 + 2) + 3 * 6;
    expect(secondsLeft(s, 1, 'exclude')).toBeCloseTo(required, 9);
    expect(secondsLeft(s, 1, 'only')).toBeCloseTo(6.5 + 5.5 + 2 * 6, 9);
    expect(secondsLeft(s, 1, 'include')).toBeCloseTo(required + 6.5 + 5.5 + 2 * 6, 9);
    expect(secondsLeft(s, FAKE_STREET.trials, 'include')).toBe(0);
    expect(roundMinutes(89)).toBe(1);
    expect(roundMinutes(29)).toBe(0);
    expect(roundMinutes(-5)).toBe(0);
  });
});

describe('saveAttempt', () => {
  it('accepted attempt: counts it, advances, saves, enqueues both records and kicks an upload', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const t0 = makeTrialRecord(s0, p0, 1);
    expectValid('TrialRecord', t0);
    services.uploader.flush.mockClear();
    const s1 = await saveAttempt(s0, t0);
    expectValid('SessionRecord', s1);
    expect(s1.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(Date.parse(s1.updated_at)).toBeGreaterThanOrEqual(Date.parse(s0.updated_at));
    expect(getState().session).toBe(s1);
    expect(services.store.trials.get(t0.trial_id)).toEqual(t0);
    expect(services.store.sessions.get(s0.session_id)?.progress).toEqual(s1.progress);
    expect(services.store.outbox.get(`trial:${t0.trial_id}`)?.status).toBe('pending');
    expect(services.store.outbox.get(`session:${s0.session_id}`)?.status).toBe('pending');
    expect(services.uploader.flush).toHaveBeenCalledTimes(1); // every save is sent at once
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

  it('skip: a trial that is not accepted moves on all the same (the choice trial answered "not sure")', async () => {
    const s0 = await createSession(STREET);
    const [p0] = plannedTrials(s0);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, p0, 1, { accepted: false }), { skip: true });
    expect(s1.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 0 });
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

describe('implicit calibration through saveAttempt (implicit_v1)', () => {
  /** a real Android / iPhone session (the engine prior applies) whose first block is upright */
  async function realSession(engine: 'blink' | 'webkit'): Promise<SessionRecord> {
    const s = onEngine(await createSession(CUSTOM), engine);
    setState({ session: s });
    return s;
  }

  it('the first complete, still, upright trial fills the sign, the magnitude and the check (Android)', async () => {
    const s0 = await realSession('blink');
    const [p0] = plannedTrials(s0);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, p0, 1, { simulated: false }));
    expect(s1.calibration).toMatchObject({ method: 'implicit_v1', gravity_sign: 1, upright_check: 'ok', notes: [] });
    expect(s1.calibration?.gravity_magnitude).toBeCloseTo(9.81, 6);
    expect(s1.calibration?.upright?.still).toBe(true);
    expect(s1.calibration?.gyro_bias_dps).toBeNull(); // in-hand means are tremor, never the gyroscope's bias
    expect(s1.calibration?.axis_check).toBeNull();
    expect(services.store.sessions.get(s0.session_id)?.calibration).toEqual(s1.calibration);
    expectValid('SessionRecord', s1);
    // the next upright trial changes nothing, and detection then uses the stored object itself
    const s2 = await saveAttempt(s1, makeTrialRecord(s0, plannedTrials(s0)[1], 1, { simulated: false, accgY: -9.81 }));
    expect(s2.calibration).toEqual(s1.calibration);
    expect(calibrationForDetection(s2)).toBe(s2.calibration);
  });

  it('an iPhone reports upright as −9.81: sign −1 is ok there; +1 is noted and not trusted', async () => {
    const ios = await realSession('webkit');
    const [p0] = plannedTrials(ios);
    const ok = await saveAttempt(ios, makeTrialRecord(ios, p0, 1, { simulated: false, accgY: -9.81 }));
    expect(ok.calibration).toMatchObject({ gravity_sign: -1, upright_check: 'ok', notes: [] });

    const other = await realSession('webkit');
    const odd = await saveAttempt(other, makeTrialRecord(other, plannedTrials(other)[0], 1, { simulated: false }));
    expect(odd.calibration).toMatchObject({ gravity_sign: 1, upright_check: 'inverted', notes: [UPRIGHT_SIGN_CONTRADICTS_ENGINE] });
    expect(calibrationForDetection(odd)?.gravity_sign).toBe(-1); // the engine's sign stays in charge
    expect(odd.calibration?.gravity_sign).toBe(1); // stored as measured (the analysis decides)
  });

  it('a simulated trial has no engine prior (the simulator is an Android phone whatever runs it)', async () => {
    const s0 = await realSession('webkit');
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, plannedTrials(s0)[0], 1, { simulated: true }));
    expect(s1.calibration).toMatchObject({ gravity_sign: 1, upright_check: 'ok', notes: [] });
  });

  it('waits for a usable upright hold: not complete, another grip detected, not upright, moving', async () => {
    const s0 = await realSession('blink');
    const [p0] = plannedTrials(s0);
    let s = await saveAttempt(s0, makeTrialRecord(s0, p0, 1, { simulated: false, status: 'no_movement' }));
    expect(s.calibration?.upright).toBeNull();
    s = await saveAttempt(s, makeTrialRecord(s0, p0, 2, { simulated: false, orientation_match: false, accgY: 2 }));
    expect(s.calibration?.upright).toBeNull(); // rejected for its grip: no upright hold, the next one counts
    const moving = makeTrialRecord(s0, p0, 3, { simulated: false });
    moving.streams.motion.accg_x = moving.streams.motion.accg_x.map((_, i) => (i % 2 === 0 ? 0.8 : -0.8));
    s = await saveAttempt(s, moving);
    expect(s.calibration?.upright).toBeNull();
    s = await saveAttempt(s, makeTrialRecord(s0, p0, 4, { simulated: false }));
    expect(s.calibration).toMatchObject({ gravity_sign: 1, upright_check: 'ok' });
  });

  it('the free (choice) trial is never an upright hold', async () => {
    const s0 = await createSession(STREET);
    const s1 = await saveAttempt(s0, makeTrialRecord(s0, plannedTrials(s0)[0], 1));
    expect(s1.calibration?.upright).toBeNull();
    // the first upright trial after it is
    const s2 = await saveAttempt(s1, makeTrialRecord(s0, plannedTrials(s0)[1], 1));
    expect(s2.calibration?.gravity_sign).toBe(1);
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

  it('attemptSummary: counts per planned trial, the ids, the latest attempt and the latest accepted one', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    const a = makeTrialRecord(s0, p0, 1, { created_at: '2026-10-23T10:00:01.000Z' });
    const b = makeTrialRecord(s0, p1, 1, { status: 'no_movement', created_at: '2026-10-23T10:00:09.000Z' });
    await saveAttempt(s0, a);
    await saveAttempt(s0, b);
    const sum = await attemptSummary(s0);
    expect([...sum.counts]).toEqual([
      [0, 1],
      [1, 1],
    ]);
    expect(sum.ids).toEqual(new Set([a.trial_id, b.trial_id]));
    expect(sum.latest?.trial_id).toBe(b.trial_id);
    expect(sum.latestAccepted?.trial_id).toBe(a.trial_id);
  });
});

describe('finishSession', () => {
  it('stamps the end, saves, enqueues, uploads and is final', async () => {
    const s0 = await createSession(CUSTOM);
    services.uploader.flush.mockClear();
    const done = await finishSession(s0, 'completed');
    expectValid('SessionRecord', done);
    expect(done.status).toBe('completed');
    expect(done.ended_at).not.toBeNull();
    expect(done.updated_at).toBe(done.ended_at);
    expect(getState().session).toBe(done);
    expect(services.store.sessions.get(s0.session_id)?.status).toBe('completed');
    expect(services.store.outbox.get(`session:${s0.session_id}`)?.status).toBe('pending');
    expect(services.uploader.flush).toHaveBeenCalledTimes(1);
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

  it('endStatusOf picks completed / ended_early from the freshest session (ended_early is schema-valid)', async () => {
    const s0 = await createSession(STREET);
    await saveAttempt(s0, makeTrialRecord(s0, plannedTrials(s0)[0], 1)); // s0 is stale from here on
    const early = await finishSession(s0, endStatusOf);
    expect(early.status).toBe('ended_early');
    expect(early.progress.next_trial_index).toBe(1);
    expectValid('SessionRecord', early);

    const s1 = await createSession(STREET);
    let s = s1;
    for (const p of plannedTrials(s1).slice(0, FAKE_STREET.required)) s = await saveAttempt(s, makeTrialRecord(s1, p, 1));
    expect((await finishSession(s1, endStatusOf)).status).toBe('completed');
  });
});

describe('revokeAccepted ("Redo last movement", "Last one was the wrong hand")', () => {
  it('re-saves the trial as not accepted, counts one accepted less and re-queues it as the next trial', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0] = plannedTrials(s0);
    const t = makeTrialRecord(s0, p0, 1);
    await saveAttempt(s0, t);
    services.uploader.flush.mockClear();
    const s1 = await revokeAccepted(s0, t.trial_id, 'wrong_hand');
    expect(s1.progress).toEqual({ next_trial_index: 0, n_attempts: 1, n_accepted: 0 });
    const stored = services.store.trials.get(t.trial_id)!;
    expect(stored.outcome).toMatchObject({ accepted: false, participant_flag: 'wrong_hand', note: REDO_REQUESTED_NOTE });
    expectValid('TrialRecord', stored);
    expect(services.store.outbox.get(`trial:${t.trial_id}`)?.status).toBe('pending');
    expect(services.uploader.flush).toHaveBeenCalled();
    expect(getState().session).toBe(s1);
    // the re-run is attempt 2 of the same planned trial
    expect(await nextAttemptNumber(s1, 0)).toBe(2);
    // Redo (no flag)
    const s2 = await saveAttempt(s1, makeTrialRecord(s0, p0, 2));
    const t2 = [...services.store.trials.values()].find((x) => x.attempt === 2)!;
    const s3 = await revokeAccepted(s2, t2.trial_id);
    expect(services.store.trials.get(t2.trial_id)?.outcome).toMatchObject({ accepted: false, participant_flag: null });
    expect(s3.progress).toEqual({ next_trial_index: 0, n_attempts: 2, n_accepted: 0 });
  });

  it('an older trial is never re-queued over later ones; refuses unknown trials and ended sessions', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    const a = makeTrialRecord(s0, p0, 1);
    await saveAttempt(s0, a);
    await saveAttempt(s0, makeTrialRecord(s0, p1, 1));
    const s1 = await revokeAccepted(s0, a.trial_id);
    expect(s1.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 1 });
    await expect(revokeAccepted(s1, 'nope')).rejects.toThrow(/no such trial/);
    await finishSession(s1, 'completed');
    await expect(revokeAccepted(s1, a.trial_id)).rejects.toThrow(/ended/);
  });
});

describe('setPosture (the "I\'m sitting" toggle)', () => {
  it('sets the protocol posture and every planned condition; nothing else changes', async () => {
    const s0 = await createSession(STREET);
    const s1 = await setPosture(s0, 'sitting');
    expect(s1.protocol.posture).toBe('sitting');
    expect(plannedTrials(s1).every((t) => t.condition.posture === 'sitting')).toBe(true);
    expect(s1.protocol.plan.map((b) => b.trials.length)).toEqual(s0.protocol.plan.map((b) => b.trials.length));
    expect(services.store.sessions.get(s0.session_id)?.protocol.posture).toBe('sitting');
    expectValid('SessionRecord', s1);
    expect(await setPosture(s1, 'sitting')).toBe(s1); // unchanged: the same object
    expect((await setPosture(s1, 'standing')).protocol.posture).toBe('standing');
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
    expect(o.calibration).toBe(s.calibration); // nothing to change (simulated, sign unknown): the stored object
    expect(o.ids.attempt).toBe(2);
    expect(o.ids.trial_id).toBe('tid');
    expect(o.simulated).toBe(true);
    expect(o.cues).toBe(cues);
    // a tap-started trial by default: ready cue, no return wait, no grip gate
    expect(o).toMatchObject({ readyCue: true, awaitReturn: false, gripGate: false });
    const again = buildTrialOptions(s, p0, 2, { source: fakeSource('simulated'), cues });
    expect([o.rng(), o.rng(), o.rng()]).toEqual([again.rng(), again.rng(), again.rng()]);
    const other = buildTrialOptions(s, p0, 3, { source: fakeSource('browser'), cues });
    expect(other.simulated).toBe(false);
    expect(other.rng()).not.toBe(buildTrialOptions(s, p0, 2, { source: fakeSource('browser'), cues }).rng());
  });

  it('buildTrialOptions passes the hands-free options; the grip gate needs a known gravity sign and a grip', async () => {
    const s0 = await createSession(STREET);
    const [choice, upright] = plannedTrials(s0);
    const deps = { source: fakeSource('browser'), cues: { play: vi.fn() } };
    const chained = { ...deps, readyCue: false, awaitReturn: true, gripGate: true };
    // simulated, before the first upright hold: no sign yet → no gate (a wrong sign must never stall GO)
    expect(buildTrialOptions(s0, upright, 1, chained)).toMatchObject({ readyCue: false, awaitReturn: true, gripGate: false });
    const android = onEngine(s0, 'blink'); // the engine prior gives the sign
    expect(buildTrialOptions(android, upright, 1, chained).gripGate).toBe(true);
    expect(buildTrialOptions(android, choice, 1, chained).gripGate).toBe(false); // never for 'free'
    expect(buildTrialOptions(android, upright, 1, deps).gripGate).toBe(false); // only when asked
    expect(buildTrialOptions(onEngine(s0, 'unknown'), upright, 1, chained).gripGate).toBe(false);
  });

  it('orientation detection does not trust a flat sign that contradicts the engine (screen-down calibration)', async () => {
    const s0 = await createSession({ ...CUSTOM, custom: TWO_TRIALS });
    const contradicted: SessionRecord = {
      ...onEngine(s0, 'blink'),
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

  it('orientation detection uses the engine sign while no sign is known (no calibration, or none measured yet)', async () => {
    const s0 = await createSession({ ...CUSTOM, custom: TWO_TRIALS });
    const skipped: SessionRecord = { ...onEngine(s0, 'webkit'), calibration: null };
    const p0 = plannedTrials(skipped)[0];
    const o = buildTrialOptions(skipped, p0, 1, { source: fakeSource('browser'), cues: { play: vi.fn() } });
    expect(o.calibration?.gravity_sign).toBe(-1); // WebKit's known sign, not the W3C default
    expect(skipped.calibration).toBeNull(); // nothing is stored
    expect(calibrationForDetection(onEngine(skipped, 'blink'))?.gravity_sign).toBe(1);
    expect(calibrationForDetection(onEngine(skipped, 'unknown'))).toBeNull();

    // the v2 session: implicit_v1, nothing measured yet
    const implicit: SessionRecord = onEngine(s0, 'webkit');
    expect(implicit.calibration?.method).toBe('implicit_v1');
    expect(calibrationForDetection(implicit)).toEqual({ ...implicit.calibration, gravity_sign: -1 });
    expect(implicit.calibration?.gravity_sign).toBeNull(); // the stored calibration is unchanged

    const noSign: SessionRecord = {
      ...skipped,
      calibration: { ...(implicit.calibration as CalibrationResult), method: undefined, notes: ['flat: moving'] },
    };
    expect(calibrationForDetection(noSign)?.gravity_sign).toBe(-1);
    expect(calibrationForDetection(noSign)?.notes).toEqual(['flat: moving']);

    const measured: SessionRecord = { ...noSign, calibration: { ...noSign.calibration!, gravity_sign: -1, notes: [] } };
    expect(calibrationForDetection(measured)).toBe(measured.calibration); // a trusted measurement passes through
  });

  it('a simulated session ignores the engine prior: its data follows the simulator, not the browser', async () => {
    const s0 = await createSession(CUSTOM); // simulated
    const onIphone: SessionRecord = { ...s0, device: { ...s0.device, engine: 'webkit' } };
    expect(calibrationForDetection(onIphone)).toBe(onIphone.calibration); // sign unknown, nothing changes
    expect(calibrationForDetection({ ...onIphone, calibration: null })).toBeNull();
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

  it('autoDecision: complete → accepted unless another grip was detected; an unknown match never rejects', async () => {
    const s = await createSession(STREET);
    const p = plannedTrials(s)[1];
    const ok = autoDecision(makeTrialRecord(s, p, 1, { accepted: false }));
    expect(ok.outcome).toMatchObject({ status: 'complete', accepted: true, participant_flag: null, note: AUTO_ACCEPT_NOTE });
    expectValid('TrialRecord', ok);
    expect(autoDecision(makeTrialRecord(s, p, 1, { orientation_match: null })).outcome.accepted).toBe(true);
    const wrong = autoDecision(makeTrialRecord(s, p, 1, { orientation_match: false }));
    expect(wrong.outcome).toMatchObject({ accepted: false, note: AUTO_REJECT_ORIENTATION_NOTE });
    const failed = makeTrialRecord(s, p, 1, { status: 'no_movement' });
    expect(autoDecision({ ...failed, outcome: { ...failed.outcome, accepted: true, note: 'x' } }).outcome).toMatchObject({
      accepted: false,
      note: 'x',
    });
  });

  it('the choice trial: saved unanswered first, then with the reported hand (hand_source reported)', async () => {
    const s = await createSession(STREET);
    const p0 = plannedTrials(s)[0];
    const rec = makeTrialRecord(s, p0, 1);
    expect(rec.condition).toMatchObject({ hand: 'left', hand_source: 'reported' }); // the plan's placeholder
    const waiting = awaitingHandReport(rec);
    expect(waiting.outcome).toMatchObject({ accepted: false, note: HAND_NOT_REPORTED_NOTE });
    const right = withReportedHand(waiting, 'right');
    expect(right.condition).toMatchObject({ hand: 'right', hand_source: 'reported' });
    expect(right.outcome).toMatchObject({ accepted: true, participant_flag: null, note: null });
    expectValid('TrialRecord', right);
    const unsure = withReportedHand(waiting, 'unsure');
    expect(unsure.condition.hand).toBe('left');
    expect(unsure.outcome).toMatchObject({ accepted: false, participant_flag: 'other_problem', note: HAND_UNSURE_NOTE });
    // an incomplete free trial cannot be accepted, whatever the answer
    const failed = makeTrialRecord(s, p0, 1, { status: 'too_long' });
    expect(withReportedHand(failed, 'left').outcome.accepted).toBe(false);
  });
});

describe('a complete attempt is saved before its review opens (tap review mode)', () => {
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
    const realCommit = services.store.commit;
    services.store.commit = vi.fn(async () => {
      throw new Error('QuotaExceededError');
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      expect(await saveForReview(rec, new Set())).toBeNull();
      expect(getState().toast?.kind).toBe('error');
    } finally {
      services.store.commit = realCommit;
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

// ───────────────────────────── the session runner (hands-free chain, end to end) ─────────────────────────────

interface FakeRun {
  planned: PlannedTrial;
  attempt: number;
  o: RecorderStart;
}

/**
 * The trial recorder as the runner sees it: start() records the run; the test ends the current run
 * with a record built for it (makeTrialRecord) — what the TrialController would deliver.
 */
function fakeRecorder() {
  const runs: FakeRun[] = [];
  const aborts: UiAbort[] = [];
  let running = false;
  return {
    runs,
    aborts,
    port: {
      start: (_s: SessionRecord, planned: PlannedTrial, attempt: number, o: RecorderStart) => {
        if (running) throw new Error('a trial is already running');
        runs.push({ planned, attempt, o });
        running = true;
      },
      abort: (reason: UiAbort) => void aborts.push(reason),
      isRunning: () => running,
    },
    get last(): FakeRun {
      return runs[runs.length - 1];
    },
    get running(): boolean {
      return running;
    },
    /** the current run ends with a record (complete with the arm's return observed, unless overridden) */
    end(
      runner: SessionRunner,
      overrides: Parameters<typeof makeTrialRecord>[3] = RETURNED,
      uiAbort: UiAbort | null = null,
      mutate?: (r: TrialRecord) => void,
    ): TrialRecord {
      const run = runs[runs.length - 1];
      if (!running) throw new Error('no trial is running');
      running = false;
      const rec = makeTrialRecord(getState().session!, run.planned, run.attempt, overrides);
      mutate?.(rec);
      runner.finished(rec, uiAbort);
      return rec;
    },
  };
}

type Recorder = ReturnType<typeof fakeRecorder>;

/** A street session on screen: the runner has read the stored attempts. */
async function streetRunner(opts: NewSessionOptions = STREET): Promise<{ session: SessionRecord; runner: SessionRunner; rec: Recorder }> {
  const session = await createSession(opts);
  const rec = fakeRecorder();
  const runner = createSessionRunner(rec.port);
  runner.enter();
  await settle();
  return { session, runner, rec };
}

/** The How-to card's tap, the free trial, the hand question (answered left): the first block's intro. */
async function throughChoice(runner: SessionRunner, rec: Recorder): Promise<TrialRecord> {
  runner.blockStart();
  const choice = rec.end(runner);
  await runner.answerHand('left');
  await settle();
  return choice;
}

/** Taps every intro, answers the hand question, completes every trial — until `stop` says so. */
async function drive(runner: SessionRunner, rec: Recorder, stop: (r: SessionRunner) => boolean): Promise<void> {
  for (let i = 0; i < 100 && !stop(runner); i++) {
    const v = runner.state().view;
    if (v.kind === 'intro') runner.blockStart();
    else if (v.kind === 'hand') await runner.answerHand('left');
    else if (v.kind === 'recording' && rec.running) rec.end(runner);
    // anything else: a save or the finish is under way
    await settle();
  }
  if (!stop(runner)) throw new Error(`stuck at ${runner.state().view.kind}`);
}

const stored = (id: string): TrialRecord => services.store.trials.get(id)!;
const sessionNow = (): SessionRecord => getState().session!;

describe('session runner: the How-to card, the free trial and the hand question', () => {
  it('the card is the first screen; its tap runs the free trial (ready beep, wake lock, no gate for free)', async () => {
    const { runner, rec } = await streetRunner();
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, pause: null, cursor: 0, attemptsReady: true, canRedo: false });
    expect(rec.runs).toHaveLength(0);
    runner.blockStart();
    expect(rec.runs).toHaveLength(1);
    expect(rec.last.planned.trial_index).toBe(0);
    expect(rec.last).toMatchObject({ attempt: 1, o: { readyCue: true, awaitReturn: false, gripGate: true, idleMs: IDLE_MS } });
    expect(runner.state().view).toMatchObject({ kind: 'recording', run: { chained: false, attempt: 1 } });
    expect(services.cues.unlock).toHaveBeenCalled();
    expect(services.wakeLock.acquire).toHaveBeenCalled();
    expect(services.cues.say).not.toHaveBeenCalled(); // nothing to announce: the participant chooses
    runner.blockStart(); // a second tap while recording does nothing
    expect(rec.runs).toHaveLength(1);
  });

  it('after the free trial the hand is asked; the answer is stored as the trial\'s hand (hand_source reported)', async () => {
    const { session, runner, rec } = await streetRunner();
    runner.blockStart();
    const choice = rec.end(runner);
    expect(runner.state().view).toMatchObject({ kind: 'hand' });
    await settle();
    expect(stored(choice.trial_id).outcome).toMatchObject({ accepted: false, note: HAND_NOT_REPORTED_NOTE });

    await runner.answerHand('right');
    const answered = stored(choice.trial_id);
    expect(answered.condition).toMatchObject({ hand: 'right', hand_source: 'reported' });
    expect(answered.outcome).toMatchObject({ status: 'complete', accepted: true, participant_flag: null, note: null });
    expectValid('TrialRecord', answered);
    expect(services.store.trials.size).toBe(1); // the same attempt, saved again
    expect(services.store.sessions.get(session.session_id)?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    // the first block's intro (one tap) — no trial starts by itself after the question
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 1, canRedo: true });
    expect(rec.runs).toHaveLength(1);
  });

  it('"Not sure" keeps the free trial as not accepted (other_problem) and moves on without a re-run', async () => {
    const { session, runner, rec } = await streetRunner();
    runner.blockStart();
    const choice = rec.end(runner);
    await runner.answerHand('unsure');
    expect(stored(choice.trial_id).outcome).toMatchObject({ accepted: false, participant_flag: 'other_problem', note: HAND_UNSURE_NOTE });
    expect(stored(choice.trial_id).condition.hand_source).toBe('reported');
    expect(services.store.sessions.get(session.session_id)?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 0 });
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 1, canRedo: false });
  });

  it('the "I\'m sitting" toggle sets the posture of the session and of every planned trial', async () => {
    const { runner } = await streetRunner();
    runner.setSitting(true);
    await settle();
    expect(sessionNow().protocol.posture).toBe('sitting');
    expect(plannedTrials(sessionNow()).every((t) => t.condition.posture === 'sitting')).toBe(true);
  });
});

describe('session runner: the hands-free chain', () => {
  it('a block intro tap names hand and grip; an accepted trial starts the next one at once (no beep, no screen)', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    services.cues.say.mockClear();
    runner.blockStart();
    // the hand and the first trial's speed (the grip is on the screen; a long sentence ran over the beeps)
    expect(services.cues.say).toHaveBeenCalledWith(`${en.common.hands.left}. Normal`, 'en');
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 1, o: { readyCue: true, awaitReturn: false, gripGate: true } });

    const t1 = rec.end(runner, RETURNED);
    expect(rec.runs).toHaveLength(3);
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, attempt: 1, o: { readyCue: false, awaitReturn: false, gripGate: true } });
    expect(runner.state()).toMatchObject({ view: { kind: 'recording', run: { chained: true, gripHint: false } }, cursor: 2 });
    // per-trial speech (Android): the speed word, said while the participant gets still
    expect(services.cues.say).toHaveBeenLastCalledWith('Fast', 'en');
    await settle();
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: true, note: AUTO_ACCEPT_NOTE });
    expect(sessionNow().progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
    expect(services.store.outbox.get(`trial:${t1.trial_id}`)?.status).toBe('pending');
  });

  it('awaitReturn: the next trial waits for the arm to go out when the return was not seen', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner, { timing: { return_onset_ms: null, return_end_ms: null } });
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, o: { readyCue: false, awaitReturn: true } });
  });

  it('hands change only at a block intro: the last trial of a block leads to the next intro, not a trial', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner); // trial 1
    rec.end(runner); // trial 2: the last of block 1 (left hand)
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 3 });
    expect(rec.runs).toHaveLength(3);
    runner.blockStart(); // block 2, right hand: the tap again plays the ready beep
    expect(rec.last).toMatchObject({ planned: { trial_index: 3, condition: { hand: 'right' } }, o: { readyCue: true, awaitReturn: false } });
  });

  it('a trial held in another grip is saved as rejected and the same trial runs again at once', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const wrong = rec.end(runner, { orientation_match: false, ...RETURNED });
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 2, o: { readyCue: false, awaitReturn: false } });
    expect(runner.state()).toMatchObject({ view: { kind: 'recording', run: { gripHint: true, attempt: 2 } }, cursor: 1 });
    await settle();
    expect(stored(wrong.trial_id).outcome).toMatchObject({ status: 'complete', accepted: false, note: AUTO_REJECT_ORIENTATION_NOTE });
    expect(sessionNow().progress).toEqual({ next_trial_index: 1, n_attempts: 2, n_accepted: 1 });
    rec.end(runner); // held right this time
    expect(rec.last.planned.trial_index).toBe(2);
  });

  it('no_movement: the reason shows, the same trial runs again by itself; a second failure in a row pauses', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const f1 = rec.end(runner, { status: 'no_movement' });
    expect(runner.state().view).toMatchObject({ kind: 'recording', run: { flash: 'no_movement', attempt: 1 } });
    await vi.advanceTimersByTimeAsync(RETRY_FLASH_MS - 1);
    expect(rec.runs).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(rec.runs).toHaveLength(3);
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 2, o: { readyCue: false, awaitReturn: false } });

    const f2 = rec.end(runner, { status: 'too_long' });
    expect(runner.state()).toMatchObject({ view: { kind: 'resume' }, pause: 'failed_twice' });
    await vi.advanceTimersByTimeAsync(RETRY_FLASH_MS * 2);
    expect(rec.runs).toHaveLength(3); // paused: nothing runs by itself
    await settle();
    expect(stored(f1.trial_id).outcome).toMatchObject({ status: 'no_movement', accepted: false });
    expect(stored(f2.trial_id).outcome).toMatchObject({ status: 'too_long', accepted: false });

    // Continue is a tap: audio and wake lock again, then the same trial at once (with its ready beep)
    services.cues.unlock.mockClear();
    services.wakeLock.acquire.mockClear();
    await runner.continueSession();
    expect(services.cues.unlock).toHaveBeenCalled();
    expect(services.wakeLock.acquire).toHaveBeenCalled();
    expect(runner.state().pause).toBeNull();
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 3, o: { readyCue: true } });
    // a success in between resets the count: one failure later only flashes
    rec.end(runner); // trial 1 accepted → trial 2
    rec.end(runner, { status: 'no_movement' });
    expect(runner.state().view).toMatchObject({ kind: 'recording', run: { flash: 'no_movement' } });
  });

  it('pause control / back / page hidden / no GO: an attempt without GO is discarded, one with GO is kept', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    // the long press: the recorder stops the trial ('user') before GO
    rec.end(runner, NO_GO, 'user');
    expect(runner.state()).toMatchObject({ view: { kind: 'resume' }, pause: 'paused' });
    await settle();
    expect(services.store.trials.size).toBe(1); // only the free trial: the attempt is discarded

    await runner.continueSession();
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 1 }); // the attempt number is free again
    // the system back button during a trial: the trial stops, the sheet opens
    expect(runner.back()).toBe(true);
    expect(rec.aborts).toEqual(['user']);
    expect(runner.state().pause).toBe('paused');
    const kept = rec.end(runner, { status: 'aborted', timing: { go_ms: 900, onset_ms: null } }, 'user');
    await settle();
    expect(stored(kept.trial_id).outcome).toMatchObject({ status: 'aborted', accepted: false });
    expect(stored(kept.trial_id).quality.warnings).toEqual(['ui_abort:user']);

    await runner.continueSession();
    runner.pageHidden();
    expect(rec.aborts).toEqual(['user', 'hidden']);
    rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'hidden');
    expect(runner.state()).toMatchObject({ view: { kind: 'resume' }, pause: 'hidden' });

    await runner.continueSession();
    rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'idle'); // 25 s without GO
    expect(runner.state().pause).toBe('idle');
    await runner.continueSession();
    rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'stalled');
    expect(runner.state().pause).toBe('stalled');
    await settle();
    expect(sessionNow().progress).toEqual({ next_trial_index: 1, n_attempts: 2, n_accepted: 1 });
  });

  it('a back press with no trial running only opens the sheet; a finished session lets back leave', async () => {
    const { runner, rec } = await streetRunner();
    expect(runner.back()).toBe(true);
    expect(rec.aborts).toEqual([]);
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, pause: 'paused' });
    await runner.continueSession(); // back to the screen under the sheet
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, pause: null });
    expect(rec.runs).toHaveLength(0);
    await runner.finishNow();
    expect(runner.back()).toBe(false);
  });

  it('a page hidden during the return of the arm keeps the complete trial and pauses', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    runner.pageHidden();
    const t1 = rec.end(runner, { timing: { return_onset_ms: null, return_end_ms: null } }, 'hidden');
    expect(runner.state()).toMatchObject({ view: { kind: 'resume' }, pause: 'hidden', cursor: 2 });
    await settle();
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: true, note: AUTO_ACCEPT_NOTE });
    await runner.continueSession(); // still inside block 1: trial 2 runs at once
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, o: { readyCue: true } });
  });

  it('a failed background save pauses the session; Continue saves it again before going on', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const realCommit = services.store.commit;
    services.store.commit = vi.fn(async () => {
      throw new Error('QuotaExceededError');
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    let t1: TrialRecord;
    try {
      t1 = rec.end(runner); // accepted, its save fails while trial 2 is already running
      await settle();
      expect(rec.aborts).toEqual(['save_failed']);
      rec.end(runner, NO_GO, 'save_failed');
      expect(runner.state()).toMatchObject({ view: { kind: 'resume' }, pause: 'save_failed' });
      await runner.continueSession(); // still failing: stays paused
      expect(runner.state().pause).toBe('save_failed');
    } finally {
      services.store.commit = realCommit;
      quiet.mockRestore();
    }
    await runner.continueSession();
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: true, note: AUTO_ACCEPT_NOTE });
    expect(sessionNow().progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
    expect(runner.state().pause).toBeNull();
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, attempt: 1 });
  });
});

describe('session runner: pause sheet redo, checkpoint, finishing', () => {
  it('"Last one was the wrong hand" takes the last accepted trial back and shows its block intro again', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const t1 = rec.end(runner); // accepted; trial 2 runs
    rec.end(runner, NO_GO, 'user'); // long press during trial 2, before GO
    expect(runner.state()).toMatchObject({ pause: 'paused', canRedo: true });

    await runner.revoke('wrong_hand');
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: false, participant_flag: 'wrong_hand', note: REDO_REQUESTED_NOTE });
    expect(sessionNow().progress).toEqual({ next_trial_index: 1, n_attempts: 2, n_accepted: 1 });
    expect(runner.state()).toMatchObject({ pause: null, view: { kind: 'intro', redo: true }, cursor: 1, canRedo: false });
    runner.blockStart(); // the intro names the hand again; trial 1 runs as attempt 2
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 2, o: { readyCue: true } });
  });

  it('"Redo last movement" works the same without a flag, also after failed attempts', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const t1 = rec.end(runner);
    rec.end(runner, { status: 'no_movement' }); // trial 2 fails once…
    await vi.advanceTimersByTimeAsync(RETRY_FLASH_MS);
    rec.end(runner, { status: 'no_movement' }); // …and twice: the sheet
    expect(runner.state()).toMatchObject({ pause: 'failed_twice', canRedo: true });
    await runner.revoke(null);
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: false, participant_flag: null, note: REDO_REQUESTED_NOTE });
    expect(runner.state()).toMatchObject({ view: { kind: 'intro', redo: true }, cursor: 1 });
  });

  it('the checkpoint after tier 3: Finish ends the session completed and uploads the session record', async () => {
    const { session, runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'checkpoint');
    expect(runner.state()).toMatchObject({ view: { kind: 'checkpoint' }, cursor: FAKE_STREET.firstOptionalTrial });
    expect(sessionNow().progress).toEqual({ next_trial_index: 7, n_attempts: 7, n_accepted: 7 });
    expect(rec.runs.every((r) => r.planned.trial_index < FAKE_STREET.firstOptionalTrial)).toBe(true);

    services.uploader.flush.mockClear();
    await runner.finishNow();
    const done = services.store.sessions.get(session.session_id)!;
    expect(done).toMatchObject({ status: 'completed' });
    expect(done.ended_at).not.toBeNull();
    expectValid('SessionRecord', done);
    expect(services.store.outbox.get(`session:${session.session_id}`)?.status).toBe('pending');
    expect(services.uploader.flush).toHaveBeenCalled(); // the session record goes out at once
    expect(getState().screen).toBe('done');
    expect(services.wakeLock.release).toHaveBeenCalled();
  });

  it('Bonus opens the optional blocks; after the last one the session ends by itself (completed)', async () => {
    const { session, runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'checkpoint');
    runner.startBonus();
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, bonus: true, cursor: FAKE_STREET.firstOptionalTrial });
    await drive(runner, rec, () => getState().screen === 'done');
    expect(rec.last.planned.trial_index).toBe(FAKE_STREET.trials - 1);
    const done = services.store.sessions.get(session.session_id)!;
    expect(done.status).toBe('completed');
    expect(done.progress).toEqual({ next_trial_index: FAKE_STREET.trials, n_attempts: FAKE_STREET.trials, n_accepted: FAKE_STREET.trials });
    // every record of the visit is valid and queued for upload
    for (const t of services.store.trials.values()) {
      expectValid('TrialRecord', t);
      expect(services.store.outbox.get(`trial:${t.trial_id}`)?.status).toBe('pending');
    }
    expect(new Set([...services.store.trials.values()].map((t) => t.condition.hand_source))).toEqual(new Set(['reported', 'instructed']));
  });

  it('Finish now before the main part is done ends the session ended_early (and is final)', async () => {
    const { session, runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner); // trial 1
    rec.end(runner, NO_GO, 'user'); // pause during trial 2
    await runner.finishNow();
    const done = services.store.sessions.get(session.session_id)!;
    expect(done.status).toBe('ended_early');
    expect(done.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
    expectValid('SessionRecord', done);
    expect(services.store.outbox.get(`session:${session.session_id}`)?.status).toBe('pending');
    expect(getState().screen).toBe('done');
    await runner.finishNow(); // a second tap changes nothing
    expect(services.store.sessions.get(session.session_id)?.ended_at).toBe(done.ended_at);
  });

  it('Continue later leaves the session resumable; entering again starts at its next trial\'s intro', async () => {
    const { session, runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner);
    rec.end(runner, NO_GO, 'user');
    await runner.later();
    expect(getState().screen).toBe('welcome');
    expect(services.store.sessions.get(session.session_id)?.status).toBe('in_progress');
    runner.dispose();

    const again = createSessionRunner(fakeRecorder().port);
    again.enter();
    await settle();
    expect(again.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 2, canRedo: true, attemptsReady: true });
    expect(again.nextAttempt(2)).toBe(1);
    expect(again.nextAttempt(1)).toBe(2);
  });

  it('entering a finished session goes to Done; a session at its checkpoint shows it again', async () => {
    const s0 = await createSession(STREET);
    await finishSession(s0, 'completed');
    const r1 = createSessionRunner(fakeRecorder().port);
    navigate('session');
    r1.enter();
    expect(getState().screen).toBe('done');

    const { runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'checkpoint');
    runner.dispose();
    const r2 = createSessionRunner(fakeRecorder().port);
    r2.enter();
    expect(r2.state().view.kind).toBe('checkpoint');
  });
});

describe('session runner: tap review mode (researcher setting review_each_trial)', () => {
  beforeEach(() => {
    setState({ settings: { ...getState().settings, review_each_trial: true } });
  });

  it('keeps the v1 flow: intro → Ready → recording → review (saved first) → Next → break → … → Done', async () => {
    const { session, runner, rec } = await streetRunner(CUSTOM);
    expect(session.settings.review_mode).toBe('tap');
    runner.blockStart();
    expect(runner.state().view.kind).toBe('ready');
    expect(rec.runs).toHaveLength(0);
    runner.ready();
    expect(rec.last).toMatchObject({ planned: { trial_index: 0 }, o: { readyCue: true, awaitReturn: false, gripGate: false, idleMs: null } });
    const t0 = rec.end(runner);
    expect(runner.state().view).toMatchObject({ kind: 'review' });
    await settle();
    expect(stored(t0.trial_id).outcome).toMatchObject({ accepted: false, note: REVIEW_INTERRUPTED_NOTE });

    await runner.decide({ kind: 'accept' });
    expect(stored(t0.trial_id).outcome).toMatchObject({ accepted: true, note: null });
    expect(runner.state()).toMatchObject({ view: { kind: 'break', fromBlock: 0 }, cursor: 1 });
    runner.nextBlock();
    runner.blockStart();
    runner.ready();
    rec.end(runner);
    await runner.decide({ kind: 'redo' }); // not accepted: Ready for the same trial
    expect(runner.state()).toMatchObject({ view: { kind: 'ready' }, cursor: 1 });
    runner.ready();
    expect(rec.last.attempt).toBe(2);
    rec.end(runner);
    await runner.decide({ kind: 'accept' });
    expect(getState().screen).toBe('done');
    expect(services.store.sessions.get(session.session_id)?.status).toBe('completed');
  });

  it('a failed attempt is kept and shown; "Try again" goes back to Ready', async () => {
    const { runner, rec } = await streetRunner(CUSTOM);
    runner.blockStart();
    runner.ready();
    const f = rec.end(runner, { status: 'no_movement' });
    expect(runner.state().view).toMatchObject({ kind: 'failed', reason: 'no_movement' });
    await settle();
    expect(runner.state().view).toMatchObject({ kind: 'failed', saving: false });
    expect(stored(f.trial_id).outcome.accepted).toBe(false);
    runner.retryFailed();
    expect(runner.state().view.kind).toBe('ready');
  });

  it('the page hidden during a trial stops it; tap mode keeps every attempt, also one without GO', async () => {
    const { runner, rec } = await streetRunner(CUSTOM);
    runner.blockStart();
    runner.ready();
    runner.pageHidden();
    expect(rec.aborts).toEqual(['hidden']);
    const t0 = rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'hidden');
    expect(runner.state().view).toMatchObject({ kind: 'failed', reason: 'interrupted' });
    await settle();
    expect(stored(t0.trial_id).outcome).toMatchObject({ status: 'interrupted', accepted: false });
    expect(stored(t0.trial_id).quality.warnings).toEqual(['ui_abort:page_hidden']);
    expect(runner.state().pause).toBeNull(); // tap mode: the failed screen says what happened
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
  const settleHistory = () => new Promise((r) => setTimeout(r, 5));
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
    await settleHistory();
  };

  it('a forward move pushes ONE guard entry; later moves replace it', () => {
    expect(fake.entries).toHaveLength(1);
    for (const screen of ['profile', 'sensors', 'setup', 'session', 'done'] as ScreenId[]) {
      navigate(screen);
      expect(fake.entries).toHaveLength(2);
      expect(fake.index).toBe(1);
      expect(fake.state).toMatchObject({ screen });
    }
  });

  it("back walks the screens' ← arrows and leaves the app only from the root (welcome ← profile ← sensors / setup)", async () => {
    for (const screen of ['profile', 'sensors'] as ScreenId[]) navigate(screen);
    for (const expected of ['profile', 'welcome'] as ScreenId[]) {
      await press();
      expect(getState().screen).toBe(expected);
      expect(fake.left).toBe(false);
    }
    expect(fake.index).toBe(0); // the root keeps no guard
    await press();
    expect(fake.left).toBe(true);
  });

  it('the researchers\' setup screen goes back to the profile', async () => {
    navigate('profile');
    navigate('setup');
    await press();
    expect(getState().screen).toBe('profile');
  });

  it('returning to the root in the app drops the guard, so back there leaves at once', async () => {
    navigate('profile');
    navigate('welcome'); // e.g. "Continue later" in a session
    await settleHistory();
    expect(fake.index).toBe(0);
    expect(getState().screen).toBe('welcome');
    await press();
    expect(fake.left).toBe(true);
  });

  it('the session screen takes the press and keeps a guard, however often back is pressed', async () => {
    const handler = vi.fn(() => true);
    const remove = setBackHandler(handler);
    try {
      navigate('profile');
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
    expect(getState().screen).toBe('profile'); // no handler: the session's ← target
  });

  it('a handler that declines (no session running) lets back go to the fallback target', async () => {
    const remove = setBackHandler(() => false);
    try {
      navigate('session');
      await press();
      expect(getState().screen).toBe('profile');
    } finally {
      remove();
    }
  });

  it("the runner is the session screen's back handler: it pauses a running session and declines a finished one", async () => {
    const { runner } = await streetRunner();
    const remove = setBackHandler(() => runner.back());
    try {
      navigate('profile');
      navigate('session');
      await press();
      expect(getState().screen).toBe('session');
      expect(runner.state().pause).toBe('paused');
      await runner.finishNow(); // → done
      expect(getState().screen).toBe('done');
      navigate('session'); // (a stale session screen)
      await press();
      expect(getState().screen).toBe('profile');
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
    navigate('profile');
    navigate('welcome');
    await settleHistory();
    fake.go(1); // the browser's forward button
    await settleHistory();
    expect(fake.index).toBe(0);
    expect(getState().screen).toBe('welcome');
  });

  it('a popstate of its own that never arrives does not swallow a later press', async () => {
    navigate('profile');
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

// ───────────────────────────── review fixes (Oct 2026) ─────────────────────────────
// fake street plan, seed 1: A = left, B = right
//   block 0 choice (trial 0) · block 1 A upright (1 natural, 2 fast) · block 2 B upright (3 natural, 4 slow)
//   block 3 B top_left (5) · block 4 A upright odd (6 tilted_touch) · bonus blocks 5 (7), 6 (8)

const UPRIGHT_LEFT = { hand: 'left', orientation: 'upright_toward' } as const;
const SIDEWAYS_RIGHT = { hand: 'right', orientation: 'top_left_toward' } as const;

describe('review fix: the intro names the hand against what the participant last did or saw', () => {
  it('introWording: another or an unknown hand → "Phone in your X hand"; same hand, new grip → "Turn the phone…"', () => {
    expect(introWording(UPRIGHT_LEFT, null)).toEqual({ line: 'phoneIn', switchHands: false, sameHand: false, handChanged: true, gripChanged: true });
    expect(introWording(UPRIGHT_LEFT, { hand: 'right', orientation: 'upright_toward', explicit: false })).toMatchObject({
      line: 'phoneIn',
      switchHands: true,
    });
    expect(introWording(SIDEWAYS_RIGHT, { hand: 'right', orientation: 'upright_toward', explicit: false })).toMatchObject({
      line: 'turnTo',
      switchHands: false,
      sameHand: true,
    });
    expect(introWording(UPRIGHT_LEFT, { hand: 'left', orientation: 'upright_toward', explicit: false }).line).toBe('keepIn');
    // after a redo / a resume: plainly named, "Switch hands!" only when the current hand is known to differ
    expect(introWording(UPRIGHT_LEFT, { hand: 'left', orientation: 'upright_toward', explicit: true })).toMatchObject({
      line: 'phoneIn',
      switchHands: false,
    });
  });

  it('Redo at a hand switch (block 3 RIGHT done, block 4 LEFT shown): block 3 comes back as "Phone in your RIGHT hand" + "Switch hands!"', async () => {
    const { runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'intro' && r.state().cursor === 6);
    // block 4's intro, worded against block 3's runs (right hand, sideways)
    expect(runner.state().introRef).toEqual({ hand: 'right', orientation: 'top_left_toward', explicit: false });
    runner.openPause('paused');
    await runner.revoke(null);
    await settle();
    const st = runner.state();
    expect(st).toMatchObject({ view: { kind: 'intro', redo: true }, cursor: 5, pause: null });
    // block 4's intro told the participant to take the phone in the LEFT hand
    expect(st.introRef).toEqual({ hand: 'left', orientation: 'upright_toward', explicit: true });
    expect(introWording(SIDEWAYS_RIGHT, st.introRef)).toMatchObject({ line: 'phoneIn', switchHands: true });
  });

  it('Redo inside a block: the same hand is named plainly, without "Switch hands!"', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart(); // block 1, left
    rec.end(runner); // trial 1 accepted
    rec.end(runner, NO_GO, 'user'); // pause before trial 2's GO
    await runner.revoke(null);
    const st = runner.state();
    expect(st).toMatchObject({ view: { kind: 'intro', redo: true }, cursor: 1 });
    expect(st.introRef).toEqual({ hand: 'left', orientation: 'upright_toward', explicit: true });
    expect(introWording(UPRIGHT_LEFT, st.introRef)).toMatchObject({ line: 'phoneIn', switchHands: false });
  });

  it('a resumed session (a fresh page) names the hand plainly and never says "Switch hands!"', async () => {
    const { runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'intro' && r.state().cursor === 3);
    await runner.later();
    runner.dispose();
    const again = createSessionRunner(fakeRecorder().port);
    again.enter();
    await settle();
    expect(again.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 3 });
    expect(again.state().introRef).toEqual({ hand: null, orientation: null, explicit: true });
    expect(introWording({ hand: 'right', orientation: 'upright_toward' }, again.state().introRef)).toMatchObject({
      line: 'phoneIn',
      switchHands: false,
    });
  });

  it('after the free trial: the reported hand is the reference ("Switch hands!" when it was the other one)', async () => {
    for (const [answer, sw] of [
      ['right', true],
      ['left', false],
      ['unsure', false],
    ] as const) {
      const { runner, rec } = await streetRunner();
      runner.blockStart();
      rec.end(runner);
      await runner.answerHand(answer);
      await settle();
      const ref = runner.state().introRef;
      expect(ref?.explicit).toBe(true);
      expect(introWording(UPRIGHT_LEFT, ref), answer).toMatchObject({ line: 'phoneIn', switchHands: sw });
    }
  });

  it('same hand, new grip (block 2 → 3): "Turn the phone…" with "Same hand: RIGHT"', async () => {
    const { runner, rec } = await streetRunner();
    await drive(runner, rec, (r) => r.state().view.kind === 'intro' && r.state().cursor === 5);
    expect(runner.state().introRef).toEqual({ hand: 'right', orientation: 'upright_toward', explicit: false });
    expect(introWording(SIDEWAYS_RIGHT, runner.state().introRef)).toMatchObject({ line: 'turnTo', sameHand: true, switchHands: false });
    // the pause sheet over the intro and back: the same intro keeps its wording
    runner.openPause('paused');
    await runner.continueSession();
    expect(runner.state().introRef).toEqual({ hand: 'right', orientation: 'upright_toward', explicit: false });
  });
});

describe('review fix: Redo / wrong hand refer to the last MOVEMENT', () => {
  it('after an auto-rejected attempt, "wrong hand" flags THAT attempt; the accepted trial before it stays accepted', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const t1 = rec.end(runner); // trial 1 accepted (done right)
    const t2a = rec.end(runner, { orientation_match: false, ...RETURNED }); // trial 2: another grip/hand
    rec.end(runner, NO_GO, 'user'); // the re-run is paused before GO
    await settle();
    expect(runner.state().redoTarget).toMatchObject({ trial_id: t2a.trial_id, trial_index: 2, accepted: false });
    await runner.revoke('wrong_hand');
    await settle();
    expect(stored(t2a.trial_id).outcome).toMatchObject({ accepted: false, participant_flag: 'wrong_hand', note: AUTO_REJECT_ORIENTATION_NOTE });
    expect(stored(t1.trial_id).outcome).toMatchObject({ accepted: true, participant_flag: null, note: AUTO_ACCEPT_NOTE });
    expect(sessionNow().progress).toEqual({ next_trial_index: 2, n_attempts: 3, n_accepted: 2 });
    // the hand is named again before the trial runs once more
    expect(runner.state()).toMatchObject({ view: { kind: 'intro', redo: true }, cursor: 2, canRedo: false });
  });

  it('the free trial before its answer (or answered "not sure") cannot be redone', async () => {
    const { runner, rec } = await streetRunner();
    runner.blockStart();
    rec.end(runner);
    await settle();
    expect(runner.state().view.kind).toBe('hand');
    runner.back();
    expect(runner.state()).toMatchObject({ pause: 'paused', canRedo: false });
  });

  it('a failed free trial is stored as never asked about: note hand_not_reported (its hand is the placeholder)', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { runner, rec } = await streetRunner();
    runner.blockStart();
    const failed = rec.end(runner, { status: 'no_movement' });
    await settle();
    expect(stored(failed.trial_id).outcome).toMatchObject({ status: 'no_movement', accepted: false, note: HAND_NOT_REPORTED_NOTE });
  });
});

describe('review fix: reading time before GO (RecorderStart.readMs)', () => {
  it('readTimeMs: a new special movement 2.5 s, a new speed 1.2 s, nothing new 0', () => {
    const c = (o: Partial<TrialCondition> = {}): TrialCondition => ({
      hand: 'left',
      orientation: 'upright_toward',
      speed: 'natural',
      start_position: 'natural',
      variant: 'standard',
      posture: 'standing',
      rep: 0,
      ...o,
    });
    expect(readTimeMs(null, c())).toBe(0);
    expect(readTimeMs(null, c({ speed: 'fast' }))).toBe(READ_SPEED_MS);
    expect(readTimeMs(c(), c({ speed: 'slow' }))).toBe(READ_SPEED_MS);
    expect(readTimeMs(c({ speed: 'slow' }), c({ speed: 'slow' }))).toBe(0);
    expect(readTimeMs(null, c({ variant: 'twist' }))).toBe(READ_SPECIAL_MS);
    expect(readTimeMs(c({ variant: 'twist' }), c({ variant: 'jerky' }))).toBe(READ_SPECIAL_MS);
    expect(readTimeMs(c({ variant: 'twist' }), c({ start_position: 'center' }))).toBe(READ_SPECIAL_MS);
    expect(readTimeMs(c({ variant: 'twist' }), c({ variant: 'twist' }))).toBe(0); // the same trial again
    expect(readTimeMs(c({ variant: 'twist' }), c())).toBe(0); // back to the standard movement
  });

  it('chained trials wait to be read; a re-run of the same trial does not; a tier-3 block waits from its first trial', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    expect(rec.last).toMatchObject({ planned: { trial_index: 1 }, o: { readMs: 0 } }); // natural, first
    rec.end(runner);
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, o: { readMs: READ_SPEED_MS } }); // → FAST
    rec.end(runner, { orientation_match: false, ...RETURNED });
    expect(rec.last).toMatchObject({ planned: { trial_index: 2 }, o: { readMs: 0 } }); // the same trial again
    await drive(runner, rec, (r) => r.state().view.kind === 'intro' && r.state().cursor === 6);
    runner.blockStart();
    expect(rec.last).toMatchObject({ planned: { trial_index: 6, condition: { variant: 'tilted_touch' } }, o: { readMs: READ_SPECIAL_MS } });
  });

  it('the words said at the intro tap: GO waits for them, + SPEECH_DUCK_TAIL_MS on an iPhone', async () => {
    services.cues.say.mockReturnValue(true); // a phone that speaks
    const android = await streetRunner();
    await throughChoice(android.runner, android.rec);
    android.runner.blockStart();
    const said = `${en.common.hands.left}. Normal`;
    expect(services.cues.say).toHaveBeenLastCalledWith(said, 'en');
    expect(android.rec.last.o.readMs).toBe(speechMs(said));

    setState({ device: { ...makeDevice(), platform: 'ios', engine: 'webkit' } });
    const iphone = await streetRunner();
    await throughChoice(iphone.runner, iphone.rec);
    iphone.runner.blockStart();
    expect(iphone.rec.last.o.readMs).toBe(speechMs(said) + SPEECH_DUCK_TAIL_MS);
  });
});

describe('review fix: screens after a missed return arm late; idle in a wrong grip; the wake lock', () => {
  it('slowArm after a trial whose arm was not seen going back out; a tap clears it', async () => {
    const { runner, rec } = await streetRunner();
    runner.blockStart();
    rec.end(runner, { timing: { return_onset_ms: null, return_end_ms: null } }); // the free trial
    expect(runner.state()).toMatchObject({ view: { kind: 'hand' }, slowArm: true });
    await runner.answerHand('left');
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, slowArm: false });
    runner.blockStart();
    rec.end(runner); // trial 1, back out
    rec.end(runner, { timing: { return_onset_ms: null, return_end_ms: null } }); // trial 2, the last of block 1
    expect(runner.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 3, slowArm: true });
  });

  it('no GO for a while because the grip gate held it back → the pause sheet says how to hold the phone', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'idle', (r) => r.quality.warnings.push(WRONG_GRIP_AT_STOP));
    expect(runner.state().pause).toBe('wrong_grip');
    await runner.continueSession();
    rec.end(runner, { ...NO_GO, status: 'interrupted' }, 'idle'); // idle for another reason
    expect(runner.state().pause).toBe('idle');
  });

  it('a real phone without the screen wake lock: records that reached GO carry wake_lock_not_held', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    services.wakeLock.active = false;
    const real = rec.end(runner, { ...RETURNED, simulated: false });
    const sim = rec.end(runner, { ...RETURNED, simulated: true });
    services.wakeLock.active = true;
    await settle();
    expect(stored(real.trial_id).quality.warnings).toContain(WAKE_LOCK_NOT_HELD);
    expect(stored(sim.trial_id).quality.warnings).not.toContain(WAKE_LOCK_NOT_HELD);
  });
});

describe('review fix: attempt numbers never repeat', () => {
  it('a failed read of the stored attempts keeps Start disabled and is retried', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    rec.end(runner, { status: 'aborted', timing: { go_ms: 900, onset_ms: null } }, 'user'); // kept: trial 1, attempt 1
    await settle();
    await runner.later();
    runner.dispose();
    const realList = services.store.listTrials.bind(services.store);
    let failures = 1;
    services.store.listTrials = vi.fn(async (sid: string) => {
      if (failures-- > 0) throw new Error('UnknownError: Connection to Indexed Database server lost');
      return realList(sid);
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const rec2 = fakeRecorder();
    const again = createSessionRunner(rec2.port);
    try {
      again.enter();
      await settle();
      expect(again.state()).toMatchObject({ view: { kind: 'intro' }, cursor: 1, attemptsReady: false });
      again.blockStart(); // not yet: an attempt number must not repeat
      expect(rec2.runs).toHaveLength(0);
      await vi.advanceTimersByTimeAsync(ATTEMPT_READ_RETRY_MS);
      await settle();
    } finally {
      quiet.mockRestore();
    }
    expect(again.state().attemptsReady).toBe(true);
    again.blockStart();
    expect(rec2.last).toMatchObject({ planned: { trial_index: 1 }, attempt: 2 });
  });

  it(`when the reads keep failing, the numbers start above ATTEMPT_FLOOR (${ATTEMPT_FLOOR})`, async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await createSession(STREET);
    services.store.listTrials = vi.fn(async () => {
      throw new Error('UnknownError: Connection to Indexed Database server lost');
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const rec = fakeRecorder();
    const runner = createSessionRunner(rec.port);
    // settle() reads the attempts itself (and would fail too): plain microtask rounds instead
    const flush = async () => {
      for (let i = 0; i < 50; i++) await Promise.resolve();
    };
    try {
      runner.enter();
      // reads at 0, 1, 2, 3 s: the first ATTEMPT_READ_TRIES − 1 failures keep Start disabled
      for (let i = 1; i < ATTEMPT_READ_TRIES; i++) {
        await flush();
        expect(runner.state().attemptsReady, `after read ${i}`).toBe(false);
        await vi.advanceTimersByTimeAsync(ATTEMPT_READ_RETRY_MS);
      }
      await flush();
    } finally {
      quiet.mockRestore();
    }
    expect(runner.state().attemptsReady).toBe(true);
    expect(runner.nextAttempt(0)).toBe(ATTEMPT_FLOOR + 1);
  });

  it('an attempt saved while the screen goes away (unmount) still counts for the attempt numbers', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart(); // trial 1, attempt 1
    // the session cannot be finished (its record cannot be written): the runner stays
    const realCommit = services.store.commit.bind(services.store);
    services.store.commit = vi.fn(async (c) => {
      if (!c.trial) throw new Error('QuotaExceededError');
      return realCommit(c);
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      const finishing = runner.finishNow();
      expect(rec.aborts).toEqual(['unmount']);
      rec.end(runner, { status: 'aborted', timing: { go_ms: 900, onset_ms: null } }, 'unmount');
      await finishing;
      await settle();
    } finally {
      services.store.commit = realCommit;
      quiet.mockRestore();
    }
    expect(runner.nextAttempt(1)).toBe(2);
  });
});

describe('review fix: atomic saves — progress survives failed and half-reported commits', () => {
  it('the last required trial: its save fails once, Continue retries, Finish → completed with every trial counted', async () => {
    const { session, runner, rec } = await streetRunner();
    const lastRequired = FAKE_STREET.required - 1;
    await drive(runner, rec, (r) => r.state().view.kind === 'recording' && rec.running && rec.last.planned.trial_index === lastRequired);
    const realCommit = services.store.commit.bind(services.store);
    let failOnce = true;
    services.store.commit = vi.fn(async (c) => {
      if (failOnce) {
        failOnce = false;
        throw new Error('UnknownError: Connection to Indexed Database server lost');
      }
      return realCommit(c);
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      rec.end(runner);
      await settle();
      expect(runner.state()).toMatchObject({ view: { kind: 'checkpoint' }, pause: 'save_failed' });
      await runner.continueSession();
      await settle();
      expect(runner.state().pause).toBeNull();
      await runner.finishNow();
    } finally {
      services.store.commit = realCommit;
      quiet.mockRestore();
    }
    const done = services.store.sessions.get(session.session_id)!;
    expect(done.status).toBe('completed');
    expect(done.progress).toEqual({ next_trial_index: FAKE_STREET.required, n_attempts: FAKE_STREET.required, n_accepted: FAKE_STREET.required });
  });

  it('a commit that was stored although its promise rejected: the retry builds on the stored session and counts once', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const realCommit = services.store.commit.bind(services.store);
    let lieOnce = true;
    services.store.commit = vi.fn(async (c) => {
      await realCommit(c);
      if (lieOnce) {
        lieOnce = false;
        throw new Error('UnknownError: Connection to Indexed Database server lost'); // after the transaction completed
      }
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      rec.end(runner); // trial 1 accepted; its save "fails" while trial 2 runs
      await settle();
      rec.end(runner, NO_GO, 'save_failed');
      expect(runner.state().pause).toBe('save_failed');
      await runner.continueSession(); // the retry
      await settle();
    } finally {
      services.store.commit = realCommit;
      quiet.mockRestore();
    }
    const expected = { next_trial_index: 2, n_attempts: 2, n_accepted: 2 };
    expect(sessionNow().progress).toEqual(expected);
    expect(services.store.sessions.get(sessionNow().session_id)?.progress).toEqual(expected);
  });

  it('revokeAccepted: a failed save leaves the trial accepted and the counters intact; the second tap re-queues it', async () => {
    const { runner, rec } = await streetRunner();
    await throughChoice(runner, rec);
    runner.blockStart();
    const t1 = rec.end(runner); // trial 1 accepted
    rec.end(runner, NO_GO, 'user'); // pause before trial 2's GO
    await settle();
    const realCommit = services.store.commit.bind(services.store);
    let failOnce = true;
    services.store.commit = vi.fn(async (c) => {
      if (failOnce) {
        failOnce = false;
        throw new Error('UnknownError: Connection to Indexed Database server lost');
      }
      return realCommit(c);
    });
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
      await runner.revoke(null); // fails: nothing written
      expect(stored(t1.trial_id).outcome.accepted).toBe(true);
      expect(runner.state()).toMatchObject({ pause: 'paused', canRedo: true });
      await runner.revoke(null); // the participant taps again
      await settle();
    } finally {
      services.store.commit = realCommit;
      quiet.mockRestore();
    }
    expect(stored(t1.trial_id).outcome.note).toBe(REDO_REQUESTED_NOTE);
    expect({ cursor: runner.state().cursor, ...sessionNow().progress }).toEqual({
      cursor: 1,
      next_trial_index: 1,
      n_attempts: 2,
      n_accepted: 1,
    });
  });

  it('finishSession counts from the stored attempts (a session saved by an older app version)', async () => {
    const s0 = await createSession(CUSTOM);
    const [p0, p1] = plannedTrials(s0);
    // both trials stored accepted, the session record never updated (the app died in between)
    await services.store.saveTrial(makeTrialRecord(s0, p0, 1));
    await services.store.saveTrial(makeTrialRecord(s0, p1, 1));
    const done = await finishSession(s0, endStatusOf);
    expect(done.status).toBe('completed');
    expect(done.progress).toEqual({ next_trial_index: 2, n_attempts: 2, n_accepted: 2 });
  });
});

describe('review fix: unfinished sessions and the outbox', () => {
  it("a new session ends the participant's other unfinished sessions as abandoned (ended_at = their last activity)", async () => {
    const old = await createSession(STREET);
    const other = await createSession(STREET);
    await settle();
    setState({ participant: makeProfile({ participant_id: 'P-XYZ234' }) });
    const someoneElse = await createSession(STREET);
    await settle();
    const closed = services.store.sessions.get(old.session_id)!;
    expect(closed).toMatchObject({ status: 'abandoned', ended_at: old.updated_at });
    expectValid('SessionRecord', closed);
    expect(services.store.outbox.get(`session:${old.session_id}`)?.status).toBe('pending');
    expect(services.store.sessions.get(other.session_id)?.status).toBe('in_progress'); // the newest one
    expect(services.store.sessions.get(someoneElse.session_id)?.status).toBe('in_progress');
    // another participant's sessions are never touched
    expect(await abandonOtherSessions(someoneElse)).toBe(0);
    expect(services.store.sessions.get(other.session_id)?.status).toBe('in_progress');
  });

  it(`recentSessionOf: the unfinished session updated within ${RESUME_ON_CONTINUE_MS / 3_600_000} h, not an older one`, async () => {
    const s = await createSession(STREET);
    expect((await recentSessionOf(s.participant_id))?.session_id).toBe(s.session_id);
    const stale = { ...services.store.sessions.get(s.session_id)!, updated_at: new Date(Date.now() - RESUME_ON_CONTINUE_MS - 60_000).toISOString() };
    services.store.sessions.set(s.session_id, stale);
    expect(await recentSessionOf(s.participant_id)).toBeNull();
  });

  it('at boot, unfinished sessions untouched for longer than they can be resumed (24 h) end as abandoned', async () => {
    const recent = await createSession(STREET);
    setState({ participant: makeProfile({ participant_id: 'P-XYZ234' }) });
    const left = await createSession(STREET);
    await settle();
    const lastSeen = new Date(Date.now() - RESUME_MAX_AGE_MS - 60_000).toISOString();
    services.store.sessions.set(left.session_id, { ...services.store.sessions.get(left.session_id)!, updated_at: lastSeen });
    expect(await abandonStaleSessions()).toBe(1);
    expect(services.store.sessions.get(left.session_id)).toMatchObject({ status: 'abandoned', ended_at: lastSeen });
    expect(services.store.outbox.get(`session:${left.session_id}`)?.status).toBe('pending');
    expect(services.store.sessions.get(recent.session_id)?.status).toBe('in_progress');
    expect(await abandonStaleSessions()).toBe(0);
  });

  it('resumeSession queues a stored trial that never got its outbox item (an older app killed between writes)', async () => {
    const s0 = await createSession(CUSTOM);
    const t = makeTrialRecord(s0, plannedTrials(s0)[0], 1);
    await services.store.saveTrial(t); // no outbox item
    expect(services.store.outbox.has(`trial:${t.trial_id}`)).toBe(false);
    setState({ session: null });
    const resumed = await resumeSession(s0.session_id);
    expect(resumed?.progress).toEqual({ next_trial_index: 1, n_attempts: 1, n_accepted: 1 });
    expect(services.store.outbox.get(`trial:${t.trial_id}`)?.status).toBe('pending');
  });
});
