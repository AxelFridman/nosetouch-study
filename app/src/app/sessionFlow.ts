/**
 * Session lifecycle: create from the setup choices, advance through the plan, save every attempt,
 * resume after reloads, finish. Persists through LocalStore and enqueues uploads.
 *
 * Rules:
 *  - createSession(): builds the plan (protocol.buildPlan with a fresh randomSeed unless given),
 *    snapshots participant (redactProfile), device, capabilities, calibration, settings
 *    (segmentation_params = DEFAULT_SEGMENTATION_PARAMS), consent {version: config.consent_version,
 *    accepted_at: state.consentAcceptedAt ?? now}; status 'in_progress'; simulated =
 *    settings.simulate. Saves, enqueues ('session'), sets state.session, returns it.
 *    preset 'custom' without a spec uses DEFAULT_CUSTOM_SPEC; protocol.custom is null for the other
 *    presets. timezone_offset_min = minutes EAST of UTC (local time = UTC + offset; Israel summer → 180).
 *    Throws when there is no participant profile or the plan is empty.
 *  - nextPlannedTrial(): the PlannedTrial at progress.next_trial_index, or null when finished.
 *  - nextAttemptNumber(): 1 + number of stored attempts with that trial_index in the session.
 *  - saveAttempt(): saves the trial, then updates progress: n_attempts += 1; if
 *    trial.outcome.accepted → n_accepted += 1 and next_trial_index += 1. updated_at = now. Saves the
 *    session, enqueues both records, kicks uploader.flush() (not awaited), sets state.session.
 *    Robustness (identical results in the normal flow):
 *      · next_trial_index becomes max(next_trial_index, trial.trial_index + 1), capped at n_planned_trials;
 *      · re-saving an attempt that is already stored (same trial_id) does not count a new attempt; only a
 *        change of outcome.accepted moves n_accepted (and next_trial_index forward on acceptance);
 *      · the base is the freshest copy of the session: state.session when it is the same session and
 *        further along, so a stale copy held by a caller can never roll progress back;
 *      · throws if the trial belongs to another session.
 *  - finishSession(): status, ended_at, updated_at; save; enqueue; flush; set state.session. A session
 *    that is already completed/abandoned is final and returned unchanged.
 *  - resumeSession(id): loads it into state.session if it is still 'in_progress' (else null). Progress
 *    is reconciled with the stored attempts first (reconcileProgress), which repairs the counters if
 *    the app was killed between saving a trial and saving the session.
 *  All mutating calls (and nextAttemptNumber) run one at a time, in call order, so a double tap or a
 *  protective save racing with a decision cannot double-count.
 *
 * IMPLEMENTED BY: session UI agent.
 */
import type {
  CalibrationResult,
  CueSink,
  CustomProtocolSpec,
  ParticipantFlag,
  PlannedBlock,
  PlannedTrial,
  Posture,
  ProtocolOptions,
  ProtocolPreset,
  SensorSource,
  SessionRecord,
  TrialControllerOptions,
  TrialIds,
  TrialLiveState,
  TrialRecord,
} from '../core/types';
import { APP_VERSION, DEFAULT_SEGMENTATION_PARAMS, PROTOCOL_VERSION, SCHEMA_VERSION } from '../core/constants';
import {
  buildPlan,
  DEFAULT_CUSTOM_SPEC,
  estimateMinutes,
  flattenPlan,
  resolveFirstHand,
  resolveParams,
} from '../core/protocol';
import { isoTime, makeRng, newId, randomSeed } from '../core/util';
import { engineGravitySign, FLAT_SIGN_CONTRADICTS_ENGINE } from '../core/calibration';
import { redactProfile } from '../core/config';
import { detectCapabilities, getDeviceInfo } from '../core/sensors';
import { getServices } from './services';
import { getState, setState } from './store';

export interface NewSessionOptions {
  preset: ProtocolPreset;
  posture: Posture;
  custom: CustomProtocolSpec | null;
  seed?: number;
}

// ───────────────────────────── serialization ─────────────────────────────

let queue: Promise<unknown> = Promise.resolve();

/** Runs tasks strictly one after another (IndexedDB read-modify-write of the session record). */
function serial<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task, task);
  queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

/** state.session when it is the same session and further along than `s` (see module doc). */
function freshest(s: SessionRecord): SessionRecord {
  const cur = getState().session;
  if (!cur || cur === s || cur.session_id !== s.session_id) return s;
  const a = cur.progress;
  const b = s.progress;
  const ahead =
    a.n_attempts > b.n_attempts ||
    (a.n_attempts === b.n_attempts && (a.n_accepted > b.n_accepted || a.next_trial_index > b.next_trial_index)) ||
    (cur.status !== 'in_progress' && s.status === 'in_progress');
  return ahead ? cur : s;
}

/** Starts an upload round without blocking the participant; refreshes the outbox counters. */
function kickUpload(): void {
  try {
    const { store, uploader } = getServices();
    store.outboxCounts().then(
      (outbox) => setState({ outbox }),
      () => undefined,
    );
    uploader.flush().then(
      (r) => setState({ outbox: r.counts }),
      () => undefined,
    );
  } catch {
    // uploads are best effort: the data is safe in IndexedDB and the queue retries on its own
  }
}

function localUtcOffsetMinutes(): number {
  const off = -new Date().getTimezoneOffset();
  return off === 0 ? 0 : off; // avoid -0
}

// ───────────────────────────── plan queries (pure) ─────────────────────────────

const flatCache = new WeakMap<PlannedBlock[], PlannedTrial[]>();

export function plannedTrials(session: SessionRecord): PlannedTrial[] {
  const plan = session.protocol.plan;
  let flat = flatCache.get(plan);
  if (!flat) {
    flat = flattenPlan(plan);
    flatCache.set(plan, flat);
  }
  return flat;
}

export function nextPlannedTrial(session: SessionRecord): PlannedTrial | null {
  const i = session.progress.next_trial_index;
  const trials = plannedTrials(session);
  return i >= 0 && i < trials.length ? trials[i] : null;
}

export function blockOf(session: SessionRecord, blockIndex: number): PlannedBlock | null {
  return session.protocol.plan.find((b) => b.block_index === blockIndex) ?? null;
}

/** 1-based position of a block in the plan and the number of blocks (for "Block i of n"). */
export function blockPosition(session: SessionRecord, blockIndex: number): { position: number; count: number } {
  const plan = session.protocol.plan;
  const i = plan.findIndex((b) => b.block_index === blockIndex);
  return { position: i < 0 ? 1 : i + 1, count: Math.max(1, plan.length) };
}

export type SessionStep = 'trial' | 'block_break' | 'finished';

/** What follows once a trial of block `acceptedBlockIndex` has been accepted and saved. */
export function stepAfterAccept(session: SessionRecord, acceptedBlockIndex: number): SessionStep {
  const next = nextPlannedTrial(session);
  if (!next) return 'finished';
  return next.block_index === acceptedBlockIndex ? 'trial' : 'block_break';
}

/** Wall-clock minutes from start to end (or to the last update while still running). */
export function sessionMinutes(session: SessionRecord): number | null {
  const start = Date.parse(session.started_at);
  const end = Date.parse(session.ended_at ?? session.updated_at);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  return Math.round((end - start) / 6000) / 10;
}

// ───────────────────────────── trial set-up (pure) ─────────────────────────────

/**
 * Deterministic uint32 seed for one attempt (drives the random foreperiod), so a trial can be
 * re-simulated exactly from (session seed, trial_index, attempt).
 */
export function trialSeed(sessionSeed: number, trialIndex: number, attempt: number): number {
  let h = (sessionSeed ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (trialIndex + 1), 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h ^ (attempt * 0x27d4eb2f), 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export function buildTrialIds(
  session: SessionRecord,
  planned: PlannedTrial,
  attempt: number,
  trialId: string = newId(),
): TrialIds {
  return {
    trial_id: trialId,
    session_id: session.session_id,
    participant_id: session.participant_id,
    study_code: session.study_code,
    block_index: planned.block_index,
    trial_index: planned.trial_index,
    trial_in_block: planned.trial_in_block,
    attempt,
  };
}

export interface TrialRunDeps {
  source: SensorSource;
  cues: CueSink;
  onLive?: (s: TrialLiveState) => void;
  getScreenOrientation?: () => { type: string | null; angle: number | null };
  isHidden?: () => boolean;
  trialId?: string;
}

/** Options for createTrialController for one attempt of a planned trial of this session. */
/**
 * The calibration the trial controller uses to classify how the phone is held. A flat step whose sign
 * contradicts the browser engine (FLAT_SIGN_CONTRADICTS_ENGINE, typically done screen-down) is not
 * trusted for that: the engine's known sign is used instead, so the review does not show a "held
 * differently" warning on every trial. session.calibration itself is stored unchanged — the analysis
 * resolves the sign from the trial data (analysis/motionlab/conventions.resolve_gravity_sign).
 */
export function calibrationForDetection(session: SessionRecord): CalibrationResult | null {
  const cal = session.calibration;
  if (!cal || !cal.notes.includes(FLAT_SIGN_CONTRADICTS_ENGINE)) return cal;
  return { ...cal, gravity_sign: engineGravitySign(session.device.engine) };
}

export function buildTrialOptions(
  session: SessionRecord,
  planned: PlannedTrial,
  attempt: number,
  deps: TrialRunDeps,
): TrialControllerOptions {
  return {
    source: deps.source,
    cues: deps.cues,
    ids: buildTrialIds(session, planned, attempt, deps.trialId),
    condition: planned.condition,
    params: resolveParams(session.settings.segmentation_params, planned.condition),
    calibration: calibrationForDetection(session),
    rng: makeRng(trialSeed(session.protocol.seed, planned.trial_index, attempt)),
    appVersion: APP_VERSION,
    // the actual source decides: a resumed session may run with other sensors than it started with
    simulated: deps.source.kind === 'simulated',
    onLive: deps.onLive,
    getScreenOrientation: deps.getScreenOrientation,
    isHidden: deps.isHidden,
  };
}

export type TrialDecision = { kind: 'accept' } | { kind: 'redo' } | { kind: 'flag'; flag: ParticipantFlag };

/** Outcome note of an attempt saved protectively because the review was left unanswered. */
export const REVIEW_INTERRUPTED_NOTE = 'review_interrupted';

/**
 * Copy of the controller's record with the participant's review decision applied (only a complete
 * trial can be accepted). A protective 'review_interrupted' note is dropped: the review was answered.
 */
export function applyDecision(record: TrialRecord, decision: TrialDecision): TrialRecord {
  const accepted = decision.kind === 'accept' && record.outcome.status === 'complete';
  return {
    ...record,
    outcome: {
      ...record.outcome,
      accepted,
      participant_flag: decision.kind === 'flag' ? decision.flag : null,
      note: record.outcome.note === REVIEW_INTERRUPTED_NOTE ? null : record.outcome.note,
    },
  };
}

/** Not-accepted copy saved when the app is hidden during the review, so the recording is never lost. */
export function asUnreviewed(record: TrialRecord): TrialRecord {
  return { ...record, outcome: { ...record.outcome, accepted: false, note: REVIEW_INTERRUPTED_NOTE } };
}

/** Failed attempts are never accepted; a UI-side reason is appended to quality.warnings. */
export function asFailedAttempt(record: TrialRecord, uiWarning: string | null): TrialRecord {
  const warnings = uiWarning ? [...record.quality.warnings, uiWarning] : record.quality.warnings;
  return {
    ...record,
    outcome: { ...record.outcome, accepted: false },
    quality: { ...record.quality, warnings },
  };
}

/**
 * Progress recomputed from the stored attempts. Returns `session` itself when nothing changes.
 * next_trial_index only moves forward (never re-runs an accepted trial, never skips back).
 */
export function reconcileProgress(session: SessionRecord, trials: TrialRecord[]): SessionRecord {
  const own = trials.filter((t) => t.session_id === session.session_id);
  const accepted = own.filter((t) => t.outcome.accepted);
  let next = session.progress.next_trial_index;
  for (const t of accepted) next = Math.max(next, t.trial_index + 1);
  next = Math.min(next, session.protocol.n_planned_trials);
  const progress = { next_trial_index: next, n_attempts: own.length, n_accepted: accepted.length };
  const p = session.progress;
  if (p.next_trial_index === progress.next_trial_index && p.n_attempts === progress.n_attempts && p.n_accepted === progress.n_accepted) {
    return session;
  }
  return { ...session, progress };
}

// ───────────────────────────── lifecycle ─────────────────────────────

export async function createSession(opts: NewSessionOptions): Promise<SessionRecord> {
  const st = getState();
  const participant = st.participant;
  if (!participant) throw new Error('createSession: no participant profile');
  const { store } = getServices();

  const seed = (opts.seed ?? randomSeed()) >>> 0;
  // deep copy: the setup screen may keep mutating its own spec object
  const custom: CustomProtocolSpec | null =
    opts.preset === 'custom' ? JSON.parse(JSON.stringify(opts.custom ?? DEFAULT_CUSTOM_SPEC)) : null;
  const protocolOpts: ProtocolOptions = { preset: opts.preset, seed, posture: opts.posture, custom };
  const plan = buildPlan(protocolOpts);
  const nPlanned = flattenPlan(plan).length;
  if (nPlanned === 0) throw new Error('createSession: the protocol has no trials');

  const now = isoTime();
  const session: SessionRecord = {
    record_type: 'session',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    session_id: newId(),
    participant_id: participant.participant_id,
    study_code: st.config.study_code,
    started_at: now,
    updated_at: now,
    ended_at: null,
    status: 'in_progress',
    participant: redactProfile(participant, st.config),
    device: st.device ?? (await getDeviceInfo()),
    capabilities: st.capabilities ?? detectCapabilities(),
    calibration: st.calibration,
    protocol: {
      preset: opts.preset,
      protocol_version: PROTOCOL_VERSION,
      seed,
      posture: opts.posture,
      first_hand: resolveFirstHand(protocolOpts),
      n_planned_trials: nPlanned,
      estimated_minutes: estimateMinutes(plan),
      custom,
      plan,
    },
    settings: {
      language: st.settings.language,
      audio: st.settings.audio,
      speech: st.settings.speech,
      segmentation_params: { ...DEFAULT_SEGMENTATION_PARAMS },
    },
    progress: { next_trial_index: 0, n_attempts: 0, n_accepted: 0 },
    consent: { version: st.config.consent_version, accepted_at: st.consentAcceptedAt ?? now },
    timezone_offset_min: localUtcOffsetMinutes(),
    simulated: st.settings.simulate,
    notes: null,
  };

  await store.saveSession(session);
  await store.enqueue('session', session.session_id, session.session_id);
  setState({ session });
  return session;
}

export function nextAttemptNumber(session: SessionRecord, trialIndex: number): Promise<number> {
  return serial(async () => {
    const trials = await getServices().store.listTrials(session.session_id);
    return 1 + trials.filter((t) => t.trial_index === trialIndex).length;
  });
}

export function saveAttempt(session: SessionRecord, trial: TrialRecord): Promise<SessionRecord> {
  return serial(async () => {
    if (trial.session_id !== session.session_id) {
      throw new Error('saveAttempt: the trial belongs to another session');
    }
    const { store } = getServices();
    const base = freshest(session);
    const previous = await store.getTrial(trial.trial_id);
    await store.saveTrial(trial);

    const p = { ...base.progress };
    const advance = () => {
      p.n_accepted += 1;
      p.next_trial_index = Math.max(p.next_trial_index, trial.trial_index + 1);
    };
    if (!previous) {
      p.n_attempts += 1;
      if (trial.outcome.accepted) advance();
    } else if (trial.outcome.accepted !== previous.outcome.accepted) {
      // same attempt saved again (e.g. decided after a protective save): only acceptance can change;
      // an un-accepted trial does not roll next_trial_index back (the participant has moved on)
      if (trial.outcome.accepted) advance();
      else p.n_accepted = Math.max(0, p.n_accepted - 1);
    }
    p.next_trial_index = Math.min(p.next_trial_index, base.protocol.n_planned_trials);

    const updated: SessionRecord = { ...base, progress: p, updated_at: isoTime() };
    await store.saveSession(updated);
    await store.enqueue('session', updated.session_id, updated.session_id);
    await store.enqueue('trial', trial.trial_id, updated.session_id);
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

export function finishSession(session: SessionRecord, status: 'completed' | 'abandoned'): Promise<SessionRecord> {
  return serial(async () => {
    const base = freshest(session);
    if (base.status !== 'in_progress') {
      if (getState().session?.session_id === base.session_id) setState({ session: base });
      return base;
    }
    const { store } = getServices();
    const now = isoTime();
    const updated: SessionRecord = { ...base, status, ended_at: now, updated_at: now };
    await store.saveSession(updated);
    await store.enqueue('session', updated.session_id, updated.session_id);
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

export function resumeSession(sessionId: string): Promise<SessionRecord | null> {
  return serial(async () => {
    const { store } = getServices();
    const stored = await store.getSession(sessionId);
    if (!stored || stored.status !== 'in_progress') return null;
    const trials = await store.listTrials(sessionId);
    let session = reconcileProgress(stored, trials);
    if (session !== stored) {
      session = { ...session, updated_at: isoTime() };
      await store.saveSession(session);
      await store.enqueue('session', session.session_id, session.session_id);
      kickUpload();
    }
    setState({ session });
    return session;
  });
}
