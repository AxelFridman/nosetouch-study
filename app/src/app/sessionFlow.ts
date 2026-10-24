/**
 * Session lifecycle: create from the setup choices, advance through the plan, save every attempt,
 * resume after reloads, finish. Persists through LocalStore and enqueues uploads.
 *
 * Rules:
 *  - createSession(): builds the plan (protocol.buildPlan with a fresh randomSeed unless given),
 *    snapshots participant (redactProfile), device, capabilities, calibration (state.calibration, i.e.
 *    a guided calibration from the researcher's calibration screen, else implicitCalibration(now):
 *    method 'implicit_v1', filled later from the first still upright trial), settings
 *    (segmentation_params = DEFAULT_SEGMENTATION_PARAMS; review_mode 'tap' when the researcher setting
 *    review_each_trial is on, else 'auto'), consent {version: config.consent_version, accepted_at:
 *    state.consentAcceptedAt ?? now}; status 'in_progress'; simulated = settings.simulate. Saves and
 *    enqueues it (one commit), kicks an upload, sets state.session, returns it; then the same
 *    participant's other unfinished sessions end as 'abandoned' (abandonOtherSessions).
 *    preset 'custom' without a spec uses DEFAULT_CUSTOM_SPEC; protocol.custom is null for the other
 *    presets. timezone_offset_min = minutes EAST of UTC (local time = UTC + offset; Israel summer → 180).
 *    Throws when there is no participant profile or the plan is empty.
 *  - nextPlannedTrial(): the PlannedTrial at progress.next_trial_index, or null when finished.
 *  - nextAttemptNumber(): 1 + number of stored attempts with that trial_index in the session.
 *  - saveAttempt(): updates progress: n_attempts += 1; if trial.outcome.accepted → n_accepted += 1
 *    and next_trial_index += 1. updated_at = now. The session's implicit calibration is updated from
 *    the trial (calibration.withUprightHold; the engine prior is dropped for a simulated trial). The
 *    trial, the session and both outbox items are written by ONE store.commit() (an IndexedDB
 *    transaction: all or nothing), then uploader.flush() is kicked (not awaited) and state.session set.
 *    Option skip: next_trial_index also moves past a trial that is NOT accepted (the choice trial
 *    answered "not sure": it is never re-run).
 *    Robustness (identical results in the normal flow):
 *      · next_trial_index becomes max(next_trial_index, trial.trial_index + 1), capped at n_planned_trials;
 *      · re-saving an attempt that is already stored (same trial_id) does not count a new attempt; only a
 *        change of outcome.accepted moves n_accepted (and next_trial_index forward on acceptance);
 *      · the base is the freshest of the caller's copy, state.session and the stored copy (storedBase),
 *        so a stale copy can never roll progress back — also after a commit that was stored although
 *        its promise rejected; with the atomic commit, a retry after a failure finds the stored trial
 *        and the stored counters in agreement, so it counts exactly once;
 *      · throws if the trial belongs to another session.
 *  - revokeAccepted(): "Redo last movement" / "Last one was the wrong hand" in the pause sheet: re-saves
 *    a stored trial as not accepted (note 'redo_requested', participant_flag as given), n_accepted − 1,
 *    and puts that planned trial back as the next one (next_trial_index = its index, only when it was
 *    the trial just before the next one: an older trial is never re-queued over later ones).
 *    flagAttempt(): "Last one was the wrong hand" about a movement that was NOT accepted (rejected for
 *    its grip, or failed): only its participant_flag is set (the planned trial runs again anyway).
 *    A failed attempt of the choice trial is never asked about: asFailedAttempt gives it the note
 *    'hand_not_reported' (its hand is the plan's placeholder, like the copy saved before the question).
 *  - finishSession(): the progress is first reconciled with the stored attempts (reconcileProgress),
 *    then status (or a function of that session that picks it, e.g. endStatusOf), ended_at,
 *    updated_at; commit; flush; set state.session. A session that has already ended is final and
 *    returned unchanged. endStatusOf(): 'completed' when every trial of the non-optional blocks is
 *    behind next_trial_index, else 'ended_early' (Finish before the main part was done).
 *  - setPosture(): the "I'm sitting" toggle of the How-to card: protocol.posture and every planned
 *    trial's condition.posture (posture never changes the plan otherwise).
 *  - resumeSession(id): loads it into state.session if it is still 'in_progress' (else null). Progress
 *    is reconciled with the stored attempts first (reconcileProgress), which repairs the counters of a
 *    session saved by an older app version (separate writes), and store.enqueueMissing() queues any
 *    stored trial that never got its outbox item.
 *  - abandonOtherSessions(): the participant's other in-progress sessions → 'abandoned' (ended_at =
 *    their last update); abandonStaleSessions() (at boot): anyone's in-progress sessions untouched for
 *    longer than RESUME_MAX_AGE_MS (no longer resumable) → 'abandoned' the same way. recentSessionOf(): the unfinished session (updated within
 *    RESUME_ON_CONTINUE_MS) that the profile's Continue resumes instead of creating a new one.
 *  All mutating calls (and nextAttemptNumber / attemptSummary) run one at a time, in call order, so a
 *  double tap, a background save racing with the next trial's save, or a protective save racing with
 *  a decision cannot double-count.
 *
 * Hands-free (auto review mode, protocol 2.0.0): autoDecision() accepts a complete trial unless its
 * grip was detected as different from the instructed one (orientation_match === false); the choice
 * trial's hand is reported by the participant afterwards (withReportedHand); the plan helpers
 * (blockChange, nextStep, progressOf, secondsLeft, endStatusOf) drive the session screen's controller
 * (sessionRunner.ts), which chains the trials, the pause sheet, the checkpoint and the finish.
 *
 * IMPLEMENTED BY: session UI agent.
 */
import type {
  CalibrationResult,
  CueSink,
  CustomProtocolSpec,
  Hand,
  OrientationId,
  ParticipantFlag,
  PlannedBlock,
  PlannedTrial,
  Posture,
  ProtocolOptions,
  ProtocolPreset,
  ReviewMode,
  SensorSource,
  SessionRecord,
  SessionStatus,
  Speed,
  TrialCondition,
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
  isOptionalBlock,
  resolveFirstHand,
  resolveParams,
} from '../core/protocol';
import { isoTime, makeRng, newId, randomSeed } from '../core/util';
import { detectionCalibration, implicitCalibration, withUprightHold } from '../core/calibration';
import { redactProfile } from '../core/config';
import { RESUME_MAX_AGE_MS } from '../core/storage';
import { detectCapabilities, getDeviceInfo } from '../core/sensors';
import { getServices } from './services';
import { getState, setState } from './store';

export interface NewSessionOptions {
  preset: ProtocolPreset;
  posture: Posture;
  custom: CustomProtocolSpec | null;
  seed?: number;
}

/** How a session ends: 'ended_early' = finished by the participant before the main part was done. */
export type EndStatus = Exclude<SessionStatus, 'in_progress'>;

// ───────────────────────────── outcome notes ─────────────────────────────

/** Outcome note of an attempt saved protectively because the review was left unanswered. */
export const REVIEW_INTERRUPTED_NOTE = 'review_interrupted';
/** auto review mode: a complete trial accepted without a review screen */
export const AUTO_ACCEPT_NOTE = 'auto_accept';
/** auto review mode: a complete trial rejected because the phone was held in another grip */
export const AUTO_REJECT_ORIENTATION_NOTE = 'auto_reject:orientation';
/** the participant asked to redo this (accepted) trial from the pause sheet */
export const REDO_REQUESTED_NOTE = 'redo_requested';
/** the choice trial, saved before the participant said which hand they used (its hand is a placeholder) */
export const HAND_NOT_REPORTED_NOTE = 'hand_not_reported';
/** the choice trial: the participant did not know which hand they used (never re-run) */
export const HAND_UNSURE_NOTE = 'hand_unsure';

/**
 * The profile's Continue resumes the participant's unfinished session updated within this long
 * (a reload, an app switch, "Exit for now"), instead of starting the visit again.
 */
export const RESUME_ON_CONTINUE_MS = 2 * 60 * 60 * 1000;

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

/**
 * The newer of two copies of the same session: an ended copy wins over a running one; then more
 * attempts; then — as many attempts — a later updated_at (a revoke or a posture change keeps the count
 * but is newer); then — same time — further along. `b` when they are not the same session.
 */
function fresher(a: SessionRecord, b: SessionRecord | null | undefined): SessionRecord {
  if (!b || b === a || b.session_id !== a.session_id) return a;
  if (b.status !== 'in_progress' && a.status === 'in_progress') return b;
  if (a.status !== 'in_progress' && b.status === 'in_progress') return a;
  const pa = a.progress;
  const pb = b.progress;
  if (pa.n_attempts !== pb.n_attempts) return pb.n_attempts > pa.n_attempts ? b : a;
  const ta = Date.parse(a.updated_at);
  const tb = Date.parse(b.updated_at);
  if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb > ta ? b : a;
  return pb.n_accepted > pa.n_accepted || pb.next_trial_index > pa.next_trial_index ? b : a;
}

/** state.session when it is the same session and newer than `s` (see fresher). */
function freshest(s: SessionRecord): SessionRecord {
  return fresher(s, getState().session);
}

/**
 * The base of a read-modify-write of the session record: the freshest of the caller's copy,
 * state.session and the STORED copy. A commit that was stored although its promise rejected (WebKit can
 * report a lost connection after the transaction completed) left the stored copy ahead of the state;
 * building on the stale state would then write old counters over it.
 */
async function storedBase(s: SessionRecord): Promise<SessionRecord> {
  const base = freshest(s);
  try {
    return fresher(base, await getServices().store.getSession(s.session_id));
  } catch {
    return base; // the read failed: the commit below fails the same way and is retried
  }
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
  return plannedAt(session, session.progress.next_trial_index);
}

/** The planned trial at a flattened-plan index, or null outside the plan. */
export function plannedAt(session: SessionRecord, index: number): PlannedTrial | null {
  const trials = plannedTrials(session);
  return index >= 0 && index < trials.length ? trials[index] : null;
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

/** The review mode of a session; sessions from before schema 1.2 were all reviewed with a tap. */
export function reviewModeOf(session: SessionRecord): ReviewMode {
  return session.settings.review_mode === 'auto' ? 'auto' : 'tap';
}

/** The choice trial: the participant picks the hand and reports it afterwards (hand_source 'reported'). */
export function isChoiceTrial(c: TrialCondition): boolean {
  return c.hand_source === 'reported';
}

/** The block before `blockIndex` in plan order (null for the first one). */
export function previousBlock(session: SessionRecord, blockIndex: number): PlannedBlock | null {
  const plan = session.protocol.plan;
  const i = plan.findIndex((b) => b.block_index === blockIndex);
  return i > 0 ? plan[i - 1] : null;
}

export interface BlockChange {
  /** the first block of the plan */
  first: boolean;
  /** another hand than the block before (always after the choice block, whose hand is the participant's) */
  handChanged: boolean;
  /** another grip than the block before (always after the choice block) */
  gripChanged: boolean;
  /** the first block after the tier-1 basics (the intro adds an encouragement) */
  afterBasics: boolean;
}

/** What changes at the intro of a block, compared with the block before it in the plan. */
export function blockChange(session: SessionRecord, blockIndex: number): BlockChange {
  const block = blockOf(session, blockIndex);
  const prev = previousBlock(session, blockIndex);
  const unknown = !prev || !block || prev.kind === 'choice';
  return {
    first: !prev,
    handChanged: unknown || prev.hand !== block.hand,
    gripChanged: unknown || prev.orientation !== block.orientation,
    afterBasics: !!prev && !!block && prev.tier === 1 && (block.tier ?? 0) > 1,
  };
}

/**
 * What the participant was last told or did before a block intro appears (the session runner keeps
 * it): the hand and grip of the last run, or of the last intro shown. null hand / grip = unknown
 * (a fresh page, the free choice). explicit: the intro must name the hand plainly whatever the
 * reference says (after a redo, a resume, the free trial).
 */
export interface IntroRef {
  hand: Hand | null;
  orientation: OrientationId | null;
  explicit: boolean;
}

/** The reference of an intro from the plan alone: the block before it (no runner, tap mode, tests). */
export function planIntroRef(session: SessionRecord, blockIndex: number): IntroRef {
  const prev = previousBlock(session, blockIndex);
  if (!prev || prev.kind === 'choice') return { hand: null, orientation: null, explicit: true };
  return { hand: prev.hand, orientation: prev.orientation, explicit: false };
}

export interface IntroWording {
  /**
   * the big line: 'phoneIn' "Phone in your LEFT hand" (another or an unknown hand, or explicit);
   * 'turnTo' "Turn the phone: top to your LEFT" (same hand, another grip); 'keepIn' (nothing changes)
   */
  line: 'phoneIn' | 'turnTo' | 'keepIn';
  /** "Switch hands!": the participant is known to hold the phone in the other hand */
  switchHands: boolean;
  /** the small "Same hand: RIGHT" line (line 'turnTo') */
  sameHand: boolean;
  handChanged: boolean;
  gripChanged: boolean;
}

/**
 * How a block intro words the hand and the grip, against what the participant was last told or did
 * (`ref`; null = nothing known). A hand switch is only announced when the current hand is known.
 */
export function introWording(block: Pick<PlannedBlock, 'hand' | 'orientation'>, ref: IntroRef | null): IntroWording {
  const r: IntroRef = ref ?? { hand: null, orientation: null, explicit: true };
  const handChanged = r.hand === null || r.hand !== block.hand;
  const gripChanged = r.orientation === null || r.orientation !== block.orientation;
  const line: IntroWording['line'] = r.explicit || handChanged ? 'phoneIn' : gripChanged ? 'turnTo' : 'keepIn';
  return {
    line,
    switchHands: r.hand !== null && r.hand !== block.hand,
    sameHand: line === 'turnTo',
    handChanged,
    gripChanged,
  };
}

/** Reading time of a speed word that changed (ms): GO waits for it (TrialControllerOptions.readMs). */
export const READ_SPEED_MS = 1200;
/** Reading time of a non-standard movement or start position (ms). */
export const READ_SPECIAL_MS = 2500;

/**
 * How long GO must wait so the participant can read the instruction of trial `c` (ms), coming from
 * the trial `prev` of the same chain (null: the first trial after a tap). A new special movement or
 * start position: READ_SPECIAL_MS; a new speed (or a first trial that is not at natural speed):
 * READ_SPEED_MS; nothing new (e.g. the same trial again): 0.
 */
export function readTimeMs(prev: TrialCondition | null, c: TrialCondition): number {
  const special = c.variant !== 'standard' || c.start_position !== 'natural';
  const sameSpecial = !!prev && prev.variant === c.variant && prev.start_position === c.start_position;
  if (special && !sameSpecial) return READ_SPECIAL_MS;
  if (prev ? prev.speed !== c.speed : c.speed !== 'natural') return READ_SPEED_MS;
  return 0;
}

/** The speeds of a block's trials, in canonical order, without repeats. */
export function blockSpeeds(block: PlannedBlock): Speed[] {
  const seen = new Set(block.trials.map((t) => t.condition.speed));
  return (['natural', 'fast', 'slow'] as Speed[]).filter((sp) => seen.has(sp));
}

export type NextStep = 'trial' | 'intro' | 'checkpoint' | 'finished';

/**
 * What the session screen shows when the next planned trial is the one at `cursor`, coming from a
 * trial of block `fromBlock` (null: not coming from a trial, e.g. after a pause or a reload):
 *  finished   — no planned trial left;
 *  checkpoint — the next trial opens an optional (bonus) block and the participant has not opted in;
 *  intro      — the next trial is in another block (hands and grips change only at a block intro);
 *  trial      — same block: the hands-free chain goes on.
 */
export function nextStep(session: SessionRecord, cursor: number, fromBlock: number | null, bonus: boolean): NextStep {
  const next = plannedAt(session, cursor);
  if (!next) return 'finished';
  const block = blockOf(session, next.block_index);
  if (block && isOptionalBlock(block) && !bonus) return 'checkpoint';
  return fromBlock !== null && next.block_index === fromBlock ? 'trial' : 'intro';
}

/** Trials of the blocks that count: the non-optional ones, plus the optional ones after opting in. */
function countedTrials(session: SessionRecord, includeOptional: boolean): PlannedTrial[] {
  return session.protocol.plan.filter((b) => includeOptional || !isOptionalBlock(b)).flatMap((b) => b.trials);
}

/** Trials of the non-optional blocks at or after `from` (default: progress.next_trial_index). */
export function requiredTrialsLeft(session: SessionRecord, from: number = session.progress.next_trial_index): number {
  return countedTrials(session, false).filter((t) => t.trial_index >= from).length;
}

/** How a session ends when the participant finishes: completed once the main part is done. */
export function endStatusOf(session: SessionRecord): 'completed' | 'ended_early' {
  return requiredTrialsLeft(session) === 0 ? 'completed' : 'ended_early';
}

/** Progress for the session screen: trials before `cursor` among the counted ones. */
export function progressOf(session: SessionRecord, cursor: number, includeOptional: boolean): { done: number; total: number } {
  const trials = countedTrials(session, includeOptional);
  return { done: trials.filter((t) => t.trial_index < cursor).length, total: trials.length };
}

/** Seconds per trial in the hands-free flow (= protocol.estimateMinutes' model), by speed. */
const TRIAL_SECONDS: Record<Speed, number> = { fast: 5.5, natural: 6.5, slow: 10 };
/** + for a trial whose start position or variant is not the default (an instruction to read) */
const READ_SECONDS = 2;
/** + per block intro still to come */
const BLOCK_CHANGE_SECONDS = 6;

/**
 * Estimated seconds left from the trial at `cursor` on (protocol.estimateMinutes' timing model, no
 * setup): the counted trials at or after it plus one block change per further block. `optional`:
 * 'include' counts the bonus blocks too, 'only' counts nothing else (the bonus' own length).
 */
export function secondsLeft(session: SessionRecord, cursor: number, optional: 'exclude' | 'include' | 'only'): number {
  const blocks = session.protocol.plan.filter((b) =>
    optional === 'only' ? isOptionalBlock(b) : optional === 'include' || !isOptionalBlock(b),
  );
  let seconds = 0;
  let changes = 0;
  const current = plannedAt(session, cursor)?.block_index ?? null;
  for (const b of blocks) {
    const left = b.trials.filter((t) => t.trial_index >= cursor);
    if (left.length === 0) continue;
    if (b.block_index !== current) changes++;
    for (const { condition: c } of left) {
      seconds += (TRIAL_SECONDS[c.speed] ?? TRIAL_SECONDS.natural) + (c.start_position !== 'natural' || c.variant !== 'standard' ? READ_SECONDS : 0);
    }
  }
  return seconds + changes * BLOCK_CHANGE_SECONDS;
}

/** Whole minutes for "about N min left" (0 = less than half a minute). */
export function roundMinutes(seconds: number): number {
  return Math.max(0, Math.round(seconds / 60));
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
  /** hands-free chain: GO only after a return movement (the previous trial ended before the arm was back) */
  awaitReturn?: boolean;
  /** play the 'ready' cue (default true; chained trials pass false) */
  readyCue?: boolean;
  /**
   * ask for the grip gate (hands-free mode). It is applied only to a non-free grip and only while the
   * detection calibration has a gravity sign: with an unknown sign the classification may be wrong and
   * a wrongly withheld GO would stall the participant.
   */
  gripGate?: boolean;
  /** no GO before this long after the start (ms): reading a new instruction / the spoken words */
  readMs?: number;
}

/**
 * The calibration the trial controller uses to classify how the phone is held: the session's
 * calibration with its sign checked against the browser engine (calibration.detectionCalibration).
 * A simulated session has no engine prior: its data follows the simulator's sign convention, not the
 * browser's (the app's simulator is an Android phone, whatever runs it). The analysis resolves the
 * sign from the trial data itself (analysis/motionlab/conventions.resolve_gravity_sign).
 */
export function calibrationForDetection(session: SessionRecord): CalibrationResult | null {
  return detectionCalibration(session.calibration, session.simulated ? null : session.device.engine, session.started_at);
}

/** Options for createTrialController for one attempt of a planned trial of this session. */
export function buildTrialOptions(
  session: SessionRecord,
  planned: PlannedTrial,
  attempt: number,
  deps: TrialRunDeps,
): TrialControllerOptions {
  const calibration = calibrationForDetection(session);
  return {
    source: deps.source,
    cues: deps.cues,
    ids: buildTrialIds(session, planned, attempt, deps.trialId),
    condition: planned.condition,
    params: resolveParams(session.settings.segmentation_params, planned.condition),
    calibration,
    rng: makeRng(trialSeed(session.protocol.seed, planned.trial_index, attempt)),
    appVersion: APP_VERSION,
    // the actual source decides: a resumed session may run with other sensors than it started with
    simulated: deps.source.kind === 'simulated',
    onLive: deps.onLive,
    getScreenOrientation: deps.getScreenOrientation,
    isHidden: deps.isHidden,
    awaitReturn: deps.awaitReturn === true,
    readyCue: deps.readyCue !== false,
    gripGate:
      deps.gripGate === true && planned.condition.orientation !== 'free' && (calibration?.gravity_sign ?? null) !== null,
    readMs: typeof deps.readMs === 'number' && deps.readMs > 0 ? deps.readMs : 0,
  };
}

export type TrialDecision = { kind: 'accept' } | { kind: 'redo' } | { kind: 'flag'; flag: ParticipantFlag };

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

/**
 * auto review mode: a complete trial is accepted ('auto_accept') unless the phone was detected in
 * another grip than instructed (orientation_match === false → not accepted, 'auto_reject:orientation';
 * the trial runs again). An unknown match (null) never rejects. Anything but a complete trial is not
 * accepted (its note is kept).
 */
export function autoDecision(record: TrialRecord): TrialRecord {
  if (record.outcome.status !== 'complete') return { ...record, outcome: { ...record.outcome, accepted: false } };
  const accepted = record.detected.orientation_match !== false;
  return {
    ...record,
    outcome: {
      ...record.outcome,
      accepted,
      participant_flag: null,
      note: accepted ? AUTO_ACCEPT_NOTE : AUTO_REJECT_ORIENTATION_NOTE,
    },
  };
}

/**
 * The choice trial, saved the moment it ends and before the hand question is answered: not accepted
 * (its hand is still the plan's placeholder), note 'hand_not_reported'.
 */
export function awaitingHandReport(record: TrialRecord): TrialRecord {
  return { ...record, outcome: { ...record.outcome, accepted: false, participant_flag: null, note: HAND_NOT_REPORTED_NOTE } };
}

/**
 * The choice trial with the participant's answer: condition.hand = the reported hand and
 * hand_source 'reported', accepted when complete; "not sure" → not accepted, flag other_problem,
 * note 'hand_unsure' (the hand stays the placeholder; the trial is not re-run: save it with skip).
 */
export function withReportedHand(record: TrialRecord, answer: Hand | 'unsure'): TrialRecord {
  if (answer === 'unsure') {
    return {
      ...record,
      condition: { ...record.condition, hand_source: 'reported' },
      outcome: { ...record.outcome, accepted: false, participant_flag: 'other_problem', note: HAND_UNSURE_NOTE },
    };
  }
  return {
    ...record,
    condition: { ...record.condition, hand: answer, hand_source: 'reported' },
    outcome: { ...record.outcome, accepted: record.outcome.status === 'complete', participant_flag: null, note: null },
  };
}

/** Not-accepted copy saved when the app is hidden during the review, so the recording is never lost. */
export function asUnreviewed(record: TrialRecord): TrialRecord {
  return { ...record, outcome: { ...record.outcome, accepted: false, note: REVIEW_INTERRUPTED_NOTE } };
}

/**
 * Failed attempts are never accepted; a UI-side reason is appended to quality.warnings. A failed attempt
 * of the choice trial is never asked about: its hand is the plan's placeholder, so it gets the note
 * 'hand_not_reported' (like the copy saved before the hand question).
 */
export function asFailedAttempt(record: TrialRecord, uiWarning: string | null): TrialRecord {
  const warnings = uiWarning ? [...record.quality.warnings, uiWarning] : record.quality.warnings;
  const note = isChoiceTrial(record.condition) && !record.outcome.note ? HAND_NOT_REPORTED_NOTE : record.outcome.note;
  return {
    ...record,
    outcome: { ...record.outcome, accepted: false, note },
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

/** The most recent of a session's stored attempts (created_at, then trial_index and attempt). */
export function latestAttempt(trials: TrialRecord[]): TrialRecord | null {
  let best: TrialRecord | null = null;
  let bestT = -Infinity;
  for (const t of trials) {
    const at = Date.parse(t.created_at);
    const v = Number.isFinite(at) ? at : -Infinity;
    if (
      !best ||
      v > bestT ||
      (v === bestT && (t.trial_index > best.trial_index || (t.trial_index === best.trial_index && t.attempt > best.attempt)))
    ) {
      best = t;
      bestT = v;
    }
  }
  return best;
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
    // a guided calibration (researcher's calibration screen) when there is one, else implicit_v1
    calibration: st.calibration ?? implicitCalibration(now),
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
      review_mode: st.settings.review_each_trial === true ? 'tap' : 'auto',
    },
    progress: { next_trial_index: 0, n_attempts: 0, n_accepted: 0 },
    consent: { version: st.config.consent_version, accepted_at: st.consentAcceptedAt ?? now },
    timezone_offset_min: localUtcOffsetMinutes(),
    simulated: st.settings.simulate,
    notes: null,
  };

  await store.commit({ session });
  setState({ session });
  kickUpload();
  // the same person's older unfinished sessions end here (abandoned): they are never resumed now
  void abandonOtherSessions(session).catch((e: unknown) => console.warn('closing older sessions failed', e));
  return session;
}

export function nextAttemptNumber(session: SessionRecord, trialIndex: number): Promise<number> {
  return serial(async () => {
    const trials = await getServices().store.listTrials(session.session_id);
    return 1 + trials.filter((t) => t.trial_index === trialIndex).length;
  });
}

/** What the session screen keeps locally: attempts per planned trial, their ids, the latest attempts. */
export interface AttemptSummary {
  counts: Map<number, number>;
  ids: Set<string>;
  latest: TrialRecord | null;
  /** the most recent accepted attempt */
  latestAccepted: TrialRecord | null;
  /** the most recent attempt with a movement (GO and an onset): what the pause sheet's Redo refers to */
  latestMoved: TrialRecord | null;
}

/** The stored attempts of a session, summarized (after every save already started). */
export function attemptSummary(session: SessionRecord): Promise<AttemptSummary> {
  return serial(async () => {
    const trials = await getServices().store.listTrials(session.session_id);
    const counts = new Map<number, number>();
    for (const t of trials) counts.set(t.trial_index, (counts.get(t.trial_index) ?? 0) + 1);
    return {
      counts,
      ids: new Set(trials.map((t) => t.trial_id)),
      latest: latestAttempt(trials),
      latestAccepted: latestAttempt(trials.filter((t) => t.outcome.accepted)),
      latestMoved: latestAttempt(trials.filter((t) => typeof t.timing.go_ms === 'number' && typeof t.timing.onset_ms === 'number')),
    };
  });
}

export interface SaveOptions {
  /** move next_trial_index past this trial even though it is not accepted (see the module doc) */
  skip?: boolean;
}

export function saveAttempt(session: SessionRecord, trial: TrialRecord, opts: SaveOptions = {}): Promise<SessionRecord> {
  return serial(async () => {
    if (trial.session_id !== session.session_id) {
      throw new Error('saveAttempt: the trial belongs to another session');
    }
    const { store } = getServices();
    const base = await storedBase(session);
    // the stored copy of this attempt and the stored session were written by ONE commit, so the
    // counters of `base` already include `previous` (a retry after a failed commit finds neither)
    const previous = await store.getTrial(trial.trial_id);

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
    if (opts.skip) p.next_trial_index = Math.max(p.next_trial_index, trial.trial_index + 1);
    p.next_trial_index = Math.min(p.next_trial_index, base.protocol.n_planned_trials);

    // implicit calibration: the first still upright hold fills it (a simulated trial has no engine prior)
    const calibration = withUprightHold(base.calibration, trial, trial.simulated ? null : base.device.engine);
    const updated: SessionRecord = { ...base, calibration, progress: p, updated_at: isoTime() };
    await store.commit({ trial, session: updated });
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

/** "Redo last movement" / "Last one was the wrong hand" (see the module doc). */
export function revokeAccepted(
  session: SessionRecord,
  trialId: string,
  flag: ParticipantFlag | null = null,
): Promise<SessionRecord> {
  return serial(async () => {
    const { store } = getServices();
    const base = await storedBase(session);
    if (base.status !== 'in_progress') throw new Error('revokeAccepted: the session has ended');
    const stored = await store.getTrial(trialId);
    if (!stored || stored.session_id !== base.session_id) throw new Error('revokeAccepted: no such trial in this session');
    const revoked: TrialRecord = {
      ...stored,
      outcome: { ...stored.outcome, accepted: false, participant_flag: flag, note: REDO_REQUESTED_NOTE },
    };
    const p = { ...base.progress };
    // the stored trial and `base` come from the same commits: an accepted copy is counted in base
    if (stored.outcome.accepted) {
      p.n_accepted = Math.max(0, p.n_accepted - 1);
      if (p.next_trial_index === stored.trial_index + 1) p.next_trial_index = stored.trial_index;
    }
    const updated: SessionRecord = { ...base, progress: p, updated_at: isoTime() };
    await store.commit({ trial: revoked, session: updated });
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

/**
 * "Last one was the wrong hand" about an attempt that was NOT accepted (auto-rejected for its grip, or
 * failed): its participant_flag is set and nothing else changes (the planned trial runs again anyway).
 */
export function flagAttempt(session: SessionRecord, trialId: string, flag: ParticipantFlag): Promise<SessionRecord> {
  return serial(async () => {
    const { store } = getServices();
    const base = await storedBase(session);
    if (base.status !== 'in_progress') throw new Error('flagAttempt: the session has ended');
    const stored = await store.getTrial(trialId);
    if (!stored || stored.session_id !== base.session_id) throw new Error('flagAttempt: no such trial in this session');
    const flagged: TrialRecord = { ...stored, outcome: { ...stored.outcome, participant_flag: flag } };
    const updated: SessionRecord = { ...base, updated_at: isoTime() };
    await store.commit({ trial: flagged, session: updated });
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

/** The "I'm sitting" toggle (see the module doc). Returns the session itself when nothing changes. */
export function setPosture(session: SessionRecord, posture: Posture): Promise<SessionRecord> {
  return serial(async () => {
    const base = await storedBase(session);
    const plan = base.protocol.plan;
    const same = base.protocol.posture === posture && plan.every((b) => b.trials.every((t) => t.condition.posture === posture));
    if (same || base.status !== 'in_progress') return base;
    const { store } = getServices();
    const updated: SessionRecord = {
      ...base,
      protocol: {
        ...base.protocol,
        posture,
        plan: plan.map((b) => ({ ...b, trials: b.trials.map((t) => ({ ...t, condition: { ...t.condition, posture } })) })),
      },
      updated_at: isoTime(),
    };
    await store.commit({ session: updated });
    setState({ session: updated });
    kickUpload();
    return updated;
  });
}

export function finishSession(
  session: SessionRecord,
  status: EndStatus | ((s: SessionRecord) => EndStatus),
): Promise<SessionRecord> {
  return serial(async () => {
    const stored = await storedBase(session);
    if (stored.status !== 'in_progress') {
      if (getState().session?.session_id === stored.session_id) setState({ session: stored });
      return stored;
    }
    const { store } = getServices();
    // the final counters (and so completed / ended_early) come from the stored attempts themselves
    let base = stored;
    try {
      base = reconcileProgress(stored, await store.listTrials(stored.session_id));
    } catch (e) {
      console.warn('the stored attempts could not be read before finishing', e);
    }
    const now = isoTime();
    const updated: SessionRecord = {
      ...base,
      status: typeof status === 'function' ? status(base) : status,
      ended_at: now,
      updated_at: now,
    };
    await store.commit({ session: updated });
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
      await store.commit({ session });
    }
    // a trial stored without its outbox item by an older app version is queued now
    const repaired = await store.enqueueMissing().catch(() => 0);
    if (session !== stored || repaired > 0) kickUpload();
    setState({ session });
    return session;
  });
}

/**
 * Ends the participant's OTHER unfinished sessions as 'abandoned' (a new session was started, so they
 * are never resumed): ended_at = their last activity (updated_at), updated_at = now. Each is
 * committed and queued for upload, so the server sees a final status. Returns how many were closed.
 */
export function abandonOtherSessions(current: SessionRecord): Promise<number> {
  return serial(async () => {
    const { store } = getServices();
    const others = (await store.listSessions(current.participant_id)).filter(
      (s) => s.session_id !== current.session_id && s.status === 'in_progress',
    );
    for (const s of others) {
      await store.commit({ session: { ...s, status: 'abandoned', ended_at: s.updated_at, updated_at: isoTime() } });
    }
    if (others.length > 0) kickUpload();
    return others.length;
  });
}

/**
 * At boot: unfinished sessions of anyone on this phone untouched for longer than `maxAgeMs` (default
 * RESUME_MAX_AGE_MS, beyond which no session is offered for resuming) end as 'abandoned' (ended_at =
 * their last activity), so the server sees a final status for visits that were simply left. Returns
 * how many were closed.
 */
export function abandonStaleSessions(maxAgeMs: number = RESUME_MAX_AGE_MS, now: number = Date.now()): Promise<number> {
  return serial(async () => {
    const { store } = getServices();
    const stale = (await store.listSessions()).filter((s) => {
      const t = Date.parse(s.updated_at);
      return s.status === 'in_progress' && Number.isFinite(t) && now - t > maxAgeMs;
    });
    for (const s of stale) {
      await store.commit({ session: { ...s, status: 'abandoned', ended_at: s.updated_at, updated_at: isoTime() } });
    }
    if (stale.length > 0) kickUpload();
    return stale.length;
  });
}

/**
 * The participant's unfinished session that the profile's Continue resumes instead of starting a new
 * one: in progress and updated within `maxAgeMs` (default RESUME_ON_CONTINUE_MS). Never throws.
 */
export async function recentSessionOf(participantId: string, maxAgeMs: number = RESUME_ON_CONTINUE_MS): Promise<SessionRecord | null> {
  try {
    return (await getServices().store.findResumableSession(participantId, maxAgeMs)) ?? null;
  } catch (e) {
    console.warn('looking up an unfinished session failed', e);
    return null;
  }
}
