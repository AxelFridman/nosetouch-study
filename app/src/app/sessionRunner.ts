/**
 * The session screen's controller: everything a participant session does between taps, written
 * without Preact so that the hands-free chain is unit-tested end to end (tests/session_flow.test.ts).
 * SessionScreen.tsx renders runner.state(), forwards the taps, the trial recorder's results, the system
 * back button and page visibility, and owns the UI timing (tap guard, arm delay, live panel).
 *
 * What is on screen is derived from the stored plan and a local cursor (the flattened-plan index of the
 * next planned trial; it starts at progress.next_trial_index, so a resumed session continues at the
 * right trial, always from a block intro).
 *
 * AUTO review mode (session.settings.review_mode 'auto': the default, hands-free; spec v2 §3):
 *  - The choice block (first block of a street plan) is introduced by the How-to card; its tap
 *    (blockStart) runs the one free trial, then the hand question (view 'hand') stores the reported hand
 *    (sessionFlow.withReportedHand: condition.hand + hand_source 'reported'); "not sure" keeps the trial
 *    as not accepted (flag other_problem) and moves on without a re-run.
 *  - Every other block starts with its intro: ONE tap (blockStart) starts it; hands and grips change
 *    only there. The tap requests the iOS motion permission (when still needed), unlocks audio, takes
 *    the wake lock and says the hand + the first trial's words (the grip is on the screen and in the
 *    animation; a long sentence would run over the first beeps).
 *  - The intro words the hand and the grip against what the participant was last told or did
 *    (state.introRef, sessionFlow.introWording): the hand of the last run, of the last intro shown, or
 *    the hand reported for the free trial — not the plan's previous block, which is wrong after a redo
 *    or a resume. After a redo, a resume and the free trial it always names the hand plainly; "Switch
 *    hands!" only when the current hand is known to be the other one.
 *  - Inside a block the trials chain with no screen in between. A complete trial is decided by
 *    sessionFlow.autoDecision: accepted ('auto_accept') → the next trial starts at once (no ready beep;
 *    awaitReturn when the trial did not see the arm go back out: trialController.returnMissed; grip
 *    gate on); orientation_match === false → saved not accepted ('auto_reject:orientation') and the
 *    same planned trial runs again at once with the grip picture. Saves run in the background; a failed
 *    save pauses the session and the record is retried on Continue (and before finishing).
 *  - Reading time: GO waits (RecorderStart.readMs) until a new instruction can be read —
 *    sessionFlow.readTimeMs (a new special movement / start position 2.5 s, a new speed 1.2 s) — and
 *    until the words spoken at the start of the trial are over (speechMs, + SPEECH_DUCK_TAIL_MS on
 *    iPhones, whose speech ducks the beeps a moment longer).
 *  - no_movement / too_long: saved (not accepted), the reason shows for RETRY_FLASH_MS, then the same
 *    planned trial runs again by itself; a second failure in a row opens the pause sheet.
 *  - Pause sheet: long press on the live panel's pause control (the recorder aborts with 'user'), the
 *    system back button, the header Pause button, page hidden, stalled sensors, no GO within IDLE_MS
 *    (reason 'wrong_grip' when the grip gate was holding GO back), two failures in a row, a failed save.
 *    An attempt that never reached GO is discarded (not saved). Continue is a tap (audio unlock + wake
 *    lock again, which iOS needs) and re-runs the current trial at once inside the block whose intro was
 *    tapped (else that block's intro); Redo last movement / Last one was the wrong hand refer to the
 *    LAST MOVEMENT (the newest saved attempt with a movement onset: state.redoTarget): an accepted one
 *    is revoked (sessionFlow.revokeAccepted) and its block intro shows again; one that was not accepted
 *    runs again anyway, so it is only flagged (wrong hand) and its intro shows again; Finish now ends
 *    the session (completed when the main part is done, else ended_early); "Exit for now" goes to the
 *    start screen, where the session can be resumed.
 *  - A screen that appears right after a trial whose arm was not seen going back out (or that ran too
 *    long) arms its buttons late (state.slowArm): the phone may still be at the face.
 *  - Optional (bonus) blocks are reached only through the checkpoint's Bonus; Finish ends there.
 *  - Speech: each chained trial says its speed word (+ the name of a non-default start position /
 *    variant) while the participant gets still, on non-iOS platforms only (iOS speaks only inside a
 *    tap). Computers are silenced by the cue sink itself (say() then reports that nothing was said).
 * TAP review mode (researcher setting review_each_trial: the v1 flow, lab use): block intro → Ready →
 *  live recording → review (Next / Redo / Problem) → … with a break screen between blocks. A complete
 *  attempt is saved the moment it ends, BEFORE the review shows, as not accepted ('review_interrupted');
 *  the decision re-saves the same trial_id. Every attempt is kept. The choice trial ends with the hand
 *  question here too; the checkpoint and the pause sheet are shared.
 *
 * Common rules:
 *  - Attempt numbers are counted locally (seeded from the store by enter() before the first tap is
 *    allowed; a failed read is retried, and after ATTEMPT_READ_TRIES failures the numbers start above
 *    ATTEMPT_FLOOR so they can never repeat a stored one), so no tap ever waits for IndexedDB. A newer
 *    save of an attempt supersedes an older failed one (the retry never writes a stale copy over the
 *    participant's answer).
 *  - Records of trials that reached GO on a real phone get the warning 'wake_lock_not_held' when the
 *    screen wake lock was not held (the screen may have dimmed or locked during the block).
 *  - Page hidden → a running trial stops; tap mode: an unanswered review is saved protectively (also
 *    when the screen goes away: dispose()).
 *  - System back button: never leaves a running session; it stops a running trial like the pause
 *    control and opens the pause sheet.
 *
 * IMPLEMENTED BY: session UI agent.
 */
import type {
  DeviceInfo,
  Hand,
  OrientationId,
  ParticipantFlag,
  PermissionState,
  PlannedBlock,
  PlannedTrial,
  SessionRecord,
  TrialCondition,
  TrialRecord,
} from '../core/types';
import { returnMissed, WRONG_GRIP_AT_STOP } from '../core/trialController';
import { requestMotionPermission } from '../core/sensors';
import { stringsFor, type Strings } from '../i18n';
import { getServices } from './services';
import { getState, navigate, setState, showToast } from './store';
import {
  applyDecision,
  asFailedAttempt,
  asUnreviewed,
  attemptSummary,
  autoDecision,
  awaitingHandReport,
  blockOf,
  endStatusOf,
  finishSession,
  flagAttempt,
  isChoiceTrial,
  nextStep,
  plannedAt,
  readTimeMs,
  reviewModeOf,
  revokeAccepted,
  saveAttempt,
  setPosture,
  withReportedHand,
  type IntroRef,
  type SaveOptions,
  type TrialDecision,
} from './sessionFlow';

/** auto mode: no GO this long after a trial started → pause (the participant is not doing the task). */
export const IDLE_MS = 25_000;
/** auto mode: how long a failed trial's reason stays on screen before the trial runs again. */
export const RETRY_FLASH_MS = 1500;
/** iOS keeps other audio (the beeps) ducked about this long after speech ends (WebKit bug 278598). */
export const SPEECH_DUCK_TAIL_MS = 1000;
/** enter(): how often the stored attempts are read before the attempt numbers fall back to the floor. */
export const ATTEMPT_READ_TRIES = 4;
/** ... how long between those reads (ms). */
export const ATTEMPT_READ_RETRY_MS = 1000;
/** After the reads failed: attempt numbers start above this, so they never repeat a stored attempt. */
export const ATTEMPT_FLOOR = 100;
/** quality warning: the screen wake lock was not held when the trial ended (the screen may have dimmed). */
export const WAKE_LOCK_NOT_HELD = 'wake_lock_not_held';

// ───────────────────────────── pure helpers ─────────────────────────────

export type UiAbort = 'user' | 'hidden' | 'stalled' | 'unmount' | 'start_failed' | 'idle' | 'save_failed';
export type FailureKind = 'no_movement' | 'too_long' | 'aborted' | 'interrupted' | 'sensor_stalled' | 'error';
/** Why the pause sheet is open (its first line). 'wrong_grip': no GO for a while because of the grip. */
export type PauseReason = 'paused' | 'hidden' | 'stalled' | 'idle' | 'wrong_grip' | 'failed_twice' | 'save_failed' | 'error';

/** Why an attempt failed, for the participant-facing message (tap mode). */
export function failureKind(record: TrialRecord | null, uiAbort: UiAbort | null): FailureKind {
  if (uiAbort === 'stalled') return 'sensor_stalled';
  if (!record) return 'error';
  switch (record.outcome.status) {
    case 'no_movement':
    case 'too_long':
    case 'aborted':
    case 'interrupted':
      return record.outcome.status;
    default:
      return 'error';
  }
}

/** quality.warnings entry recording why the UI aborted the trial. */
export function uiAbortWarning(reason: UiAbort | null): string | null {
  if (!reason) return null;
  const name = reason === 'hidden' ? 'page_hidden' : reason === 'stalled' ? 'sensor_stall' : reason;
  return `ui_abort:${name}`;
}

/** The pause sheet's reason after a trial that ended with this UI abort reason (null: no normal end). */
export function pauseReasonOf(uiAbort: UiAbort | null): PauseReason {
  switch (uiAbort) {
    case 'user':
      return 'paused';
    case 'hidden':
      return 'hidden';
    case 'stalled':
      return 'stalled';
    case 'idle':
      return 'idle';
    case 'save_failed':
      return 'save_failed';
    default:
      return 'error';
  }
}

/** An attempt that reached GO is data and is kept even when stopped; an earlier stop is discarded (auto). */
export function reachedGo(record: TrialRecord | null): record is TrialRecord {
  return !!record && typeof record.timing.go_ms === 'number';
}

/** The attempt contains a movement (an onset after GO): what "Redo last movement" refers to. */
export function hasMovement(record: TrialRecord | null): record is TrialRecord {
  return reachedGo(record) && typeof record.timing.onset_ms === 'number';
}

/**
 * The phone may still be at the face when this attempt's record arrives: a complete trial whose return
 * of the arm was not seen, or a movement that ran too long. The next tap screen arms late.
 */
export function armLate(record: TrialRecord | null): boolean {
  if (!record) return false;
  return returnMissed(record) || record.outcome.status === 'too_long';
}

/** Speech outside a tap (chained trials): never on iOS / WebKit, which speaks only inside a gesture. */
export function perTrialSpeechAllowed(device: DeviceInfo | null | undefined): boolean {
  return !!device && device.platform !== 'ios' && device.engine !== 'webkit';
}

/** Rough length of an utterance (ms): about 75 ms per character after a 250 ms start. */
export function speechMs(text: string | null | undefined): number {
  return text ? 250 + 75 * text.length : 0;
}

/** What is said for one trial: the speed word, plus a non-default start position / variant name. */
export function trialWords(c: TrialCondition, s: Strings): string[] {
  const words = [s.session.speech.speeds[c.speed]];
  if (c.start_position !== 'natural') words.push(s.common.startPositions[c.start_position].label);
  if (c.variant !== 'standard') words.push(s.common.variants[c.variant].label);
  return words;
}

/**
 * What the block-intro tap says: the hand and the first trial's words (nothing for the choice trial).
 * The grip is on the screen and in the animation; saying it too made the sentence run over the first
 * trial's beeps.
 */
export function introWords(block: PlannedBlock | null, first: PlannedTrial, s: Strings): string[] {
  if (isChoiceTrial(first.condition)) return [];
  const hand = block?.hand ?? first.condition.hand;
  return [s.common.hands[hand], ...trialWords(first.condition, s)];
}

// ───────────────────────────── saving ─────────────────────────────

/** Saves an attempt into its own session; never throws (a toast reports failures). */
export async function persistAttempt(record: TrialRecord, opts?: SaveOptions): Promise<SessionRecord | null> {
  try {
    const cur = getState().session;
    const base =
      cur && cur.session_id === record.session_id ? cur : await getServices().store.getSession(record.session_id);
    if (!base) throw new Error(`session ${record.session_id} not found`);
    return await saveAttempt(base, record, opts);
  } catch (e) {
    console.error('saving the attempt failed', e);
    showToast(stringsFor(getState().settings.language).session.saveFailed, 'error');
    return null;
  }
}

/**
 * tap mode: saves a complete attempt the moment it ends, before its review shows: not accepted,
 * outcome.note 'review_interrupted'. The decision re-saves the same trial_id later (saveAttempt counts
 * that as a change of acceptance, never as another attempt). Marks it settled, so the protective save
 * on pagehide/unmount has nothing left to do.
 */
export function saveForReview(record: TrialRecord, settled: Set<string>): Promise<SessionRecord | null> {
  settled.add(record.trial_id);
  return persistAttempt(asUnreviewed(record));
}

// ───────────────────────────── side effects of taps ─────────────────────────────

function unlockAudio(): void {
  try {
    void getServices().cues.unlock().catch(() => undefined);
  } catch {
    // audio cues are optional; the visual cues still work
  }
}

function acquireWakeLock(): void {
  try {
    void getServices().wakeLock.acquire().catch(() => false);
  } catch {
    // not fatal: the screen may dim between trials
  }
}

function releaseWakeLock(): void {
  try {
    void getServices().wakeLock.release().catch(() => undefined);
  } catch {
    // nothing to release
  }
}

/** The screen wake lock is held (true when it cannot be told). */
function wakeLockHeld(): boolean {
  try {
    return getServices().wakeLock.active !== false;
  } catch {
    return true;
  }
}

/**
 * Speaks `parts` when speech is enabled; returns the text when it was actually queued (null when
 * speech is off, unavailable or silenced). On iOS it only works synchronously inside a tap handler
 * (timers/effects stay silent); callers outside a tap check perTrialSpeechAllowed first.
 */
function speak(parts: (string | null | false | undefined)[]): string | null {
  try {
    const { cues } = getServices();
    const { settings } = getState();
    if (!settings.speech || !cues.say) return null;
    const text = parts.filter(Boolean).join('. ');
    if (!text) return null;
    return cues.say(text, settings.language) === true ? text : null;
  } catch {
    // speech is a convenience; the screen and the beeps carry the protocol
    return null;
  }
}

/** iOS motion permission, requested synchronously inside a tap when still needed (null: not needed). */
function requestPermissionInTap(): Promise<PermissionState> | null {
  const st0 = getState();
  const caps = st0.capabilities;
  if (st0.settings.simulate || !caps?.permission_api || caps.permission_state === 'granted') return null;
  try {
    return requestMotionPermission();
  } catch (e) {
    console.warn('motion permission request failed', e);
    return null;
  }
}

// ───────────────────────────── the runner ─────────────────────────────

/** How the session screen starts one trial (SessionScreen's useTrialRecorder). */
export interface RecorderStart {
  /** play the 'ready' cue (taps: yes; chained trials: no) */
  readyCue: boolean;
  /** GO only after a return movement (the previous trial ended before the arm was back out) */
  awaitReturn: boolean;
  /** hands-free grip gate (applied by sessionFlow.buildTrialOptions only where it makes sense) */
  gripGate: boolean;
  /** abort('idle') when no GO came this long after the start; null = never */
  idleMs: number | null;
  /** no GO before this long after the start (ms): reading the new instruction / the speech (0: none) */
  readMs: number;
}

/** The trial recorder as the runner sees it: one TrialController at a time. */
export interface RecorderPort {
  start(session: SessionRecord, planned: PlannedTrial, attempt: number, o: RecorderStart): void;
  abort(reason: UiAbort): void;
  /** a controller is running (true at once after start(), false at once after its result) */
  isRunning(): boolean;
}

/** One run of the live panel: the planned trial, its attempt number and how it was started. */
export interface RunInfo {
  planned: PlannedTrial;
  attempt: number;
  /** started by itself (no tap): no ready beep, and the hold status shows at once */
  chained: boolean;
  /** re-run after a trial in the wrong grip: the grip picture shows until GO */
  gripHint: boolean;
  /** auto mode: the failed trial's reason, shown for RETRY_FLASH_MS before the trial runs again */
  flash: 'no_movement' | 'too_long' | null;
}

export type View =
  | { kind: 'intro'; notice?: 'permission_denied'; redo?: boolean } // block intro / the How-to card
  | { kind: 'ready'; notice?: 'permission_denied' } // tap mode
  | { kind: 'recording'; run: RunInfo }
  | { kind: 'resume' } // auto: the chain was paused; Continue runs the current trial at once
  | { kind: 'hand'; record: TrialRecord }
  | { kind: 'checkpoint' }
  | { kind: 'review'; record: TrialRecord } // tap mode
  | { kind: 'failed'; reason: FailureKind; saving: boolean } // tap mode
  | { kind: 'break'; fromBlock: number }; // tap mode

/** The movement the pause sheet's Redo / wrong hand refer to. */
export interface RedoTarget {
  trial_id: string;
  trial_index: number;
  accepted: boolean;
  condition: TrialCondition;
}

export interface RunnerState {
  view: View;
  /** the pause sheet is open (over the view), and why */
  pause: PauseReason | null;
  /** flattened-plan index of the next planned trial */
  cursor: number;
  /** the participant opted into the optional (bonus) blocks */
  bonus: boolean;
  /** a save or the finish is under way: the buttons wait */
  busy: boolean;
  /** the stored attempts were read (attempt numbers are known): the intro may start */
  attemptsReady: boolean;
  /** there is a movement that "Redo last movement" / "wrong hand" can refer to (redoTarget) */
  canRedo: boolean;
  /** the last movement (see the module doc); null: none */
  redoTarget: RedoTarget | null;
  /** an intro view: what the participant was last told or did (sessionFlow.introWording) */
  introRef: IntroRef | null;
  /** the view appeared right after a trial whose arm may still be at the face: arm its buttons late */
  slowArm: boolean;
}

export interface SessionRunner {
  /** a snapshot of what the screen shows */
  state(): RunnerState;
  /** the attempt number the next run of a planned trial gets */
  nextAttempt(trialIndex: number): number;
  /** the screen appeared: read the stored attempts; a finished session goes to Done */
  enter(): void;
  /** the screen went away: timers stop, an unanswered review is saved, the wake lock is released */
  dispose(): void;
  /** the recorder's result for the run it was started with (record null: no record) */
  finished(record: TrialRecord | null, uiAbort: UiAbort | null, error?: unknown): void;
  /** block intro / How-to card tap (auto: starts the block's first trial; tap mode: the Ready screen) */
  blockStart(): void;
  /** tap mode: the Ready tap */
  ready(): void;
  /** tap mode: the review decision */
  decide(decision: TrialDecision): Promise<void>;
  /** tap mode: "Try again" after a failed attempt */
  retryFailed(): void;
  /** tap mode: the break screen's Continue */
  nextBlock(): void;
  /** the hand question after the choice trial */
  answerHand(answer: Hand | 'unsure'): Promise<void>;
  /** pause sheet: Continue */
  continueSession(): Promise<void>;
  /** pause sheet: Redo last movement (null) / Last one was the wrong hand ('wrong_hand') */
  revoke(flag: ParticipantFlag | null): Promise<void>;
  /** pause sheet's Finish now and the checkpoint's Finish */
  finishNow(): Promise<void>;
  /** pause sheet: Exit for now (the start screen; the session stays resumable) */
  later(): Promise<void>;
  /** checkpoint: the bonus blocks */
  startBonus(): void;
  /** How-to card: the "I'm sitting" toggle */
  setSitting(sitting: boolean): void;
  /** the header Pause button */
  openPause(reason?: PauseReason): void;
  /** the system back button: true when it was handled (the app stays) */
  back(): boolean;
  /** the page was hidden (visibilitychange → hidden, pagehide) */
  pageHidden(): void;
}

/** The last instruction the participant followed or saw (see introRef). free: the free choice. */
interface Instruction {
  hand: Hand | null;
  orientation: OrientationId | null;
  free: boolean;
}

/**
 * Creates the controller of one visit of the session screen (for state.session at creation time).
 * `onChange` is called after every change of state() (the screen re-renders).
 */
export function createSessionRunner(recorder: RecorderPort, onChange: () => void = () => undefined): SessionRunner {
  const st: RunnerState = {
    view: { kind: 'intro' },
    pause: null,
    cursor: getState().session?.progress.next_trial_index ?? 0,
    bonus: false,
    busy: false,
    attemptsReady: false,
    canRedo: false,
    redoTarget: null,
    introRef: { hand: null, orientation: null, explicit: true },
    slowArm: false,
  };
  let disposed = false;

  // bookkeeping read by callbacks that outlive a render
  const attempts = { counts: new Map<number, number>(), ids: new Set<string>() };
  /** after the stored attempts could not be read: every attempt number starts above ATTEMPT_FLOOR */
  let attemptFloor = 0;
  let lastMoved: RedoTarget | null = null;
  /** saves that failed, oldest first (one per attempt: its newest copy) */
  let unsaved: { record: TrialRecord; opts?: SaveOptions }[] = [];
  /** per attempt: how many saves were started (a failed older save must not shadow a newer one) */
  const generation = new Map<string, number>();
  let run: RunInfo | null = null;
  /** the condition of the last run started (the reading time of a chained trial compares with it) */
  let lastRun: TrialCondition | null = null;
  /** the block whose intro was tapped: inside it the chain needs no further tap */
  let chainBlock: number | null = null;
  /** what the participant last followed or saw (null: nothing on this page yet) */
  let instructed: Instruction | null = null;
  /** the cursor whose intro introRef was computed for */
  let introCursor: number | null = null;
  let failStreak = 0;
  let flashTimer: ReturnType<typeof setTimeout> | undefined;
  let summaryTimer: ReturnType<typeof setTimeout> | undefined;
  /** tap mode: reviewed attempts already saved */
  const settled = new Set<string>();
  let deciding = false;
  let finishing = false;

  const set = (patch: Partial<RunnerState>): void => {
    Object.assign(st, patch);
    if (!disposed) onChange();
  };
  const goTo = (view: View): void => set({ view });
  const openPause = (reason: PauseReason): void => set({ pause: reason });
  const strings = (): Strings => stringsFor(getState().settings.language);
  const autoMode = (): boolean => {
    const cur = getState().session;
    return cur ? reviewModeOf(cur) === 'auto' : true;
  };
  const nextAttempt = (trialIndex: number): number => Math.max(attempts.counts.get(trialIndex) ?? 0, attemptFloor) + 1;

  /**
   * Shows the intro of the trial at the cursor. A new intro (another cursor than the intro on screen,
   * or a redo) words itself against what the participant last followed or saw; from then on, the intro
   * itself is that instruction.
   */
  const showIntro = (extra: { redo?: boolean; notice?: 'permission_denied' } = {}): void => {
    const cur = getState().session;
    const planned = cur ? plannedAt(cur, st.cursor) : null;
    let ref = st.introRef;
    if (extra.redo || st.view.kind !== 'intro' || introCursor !== st.cursor) {
      ref = {
        hand: instructed?.hand ?? null,
        orientation: instructed?.orientation ?? null,
        explicit: !!extra.redo || instructed === null || instructed.free,
      };
      introCursor = st.cursor;
      const block = cur && planned ? blockOf(cur, planned.block_index) : null;
      if (planned && !isChoiceTrial(planned.condition) && block?.kind !== 'choice') {
        instructed = { hand: block?.hand ?? planned.condition.hand, orientation: block?.orientation ?? planned.condition.orientation, free: false };
      }
    }
    set({ view: { kind: 'intro', ...extra }, introRef: ref });
  };

  /** Local bookkeeping of a saved attempt: attempt numbers and the pause sheet's "last movement". */
  const noteSaved = (record: TrialRecord): void => {
    if (!attempts.ids.has(record.trial_id)) {
      attempts.ids.add(record.trial_id);
      attempts.counts.set(record.trial_index, nextAttempt(record.trial_index));
    }
    if (hasMovement(record)) {
      lastMoved = { trial_id: record.trial_id, trial_index: record.trial_index, accepted: record.outcome.accepted, condition: record.condition };
    }
    set({ canRedo: redoAllowed(), redoTarget: lastMoved });
  };

  /** A choice trial that was not accepted (no answer yet, or "not sure") is never redone. */
  const redoAllowed = (): boolean => !!lastMoved && (lastMoved.accepted || !isChoiceTrial(lastMoved.condition));

  /** Saves an attempt; a failure keeps its newest copy for retryUnsaved. */
  const persist = (record: TrialRecord, opts?: SaveOptions): Promise<SessionRecord | null> => {
    const g = (generation.get(record.trial_id) ?? 0) + 1;
    generation.set(record.trial_id, g);
    unsaved = unsaved.filter((u) => u.record.trial_id !== record.trial_id);
    return persistAttempt(record, opts).then((saved) => {
      if (!saved && generation.get(record.trial_id) === g) unsaved.push({ record, opts });
      return saved;
    });
  };

  /** Retries the failed saves in order; false while one still fails. */
  const retryUnsaved = async (): Promise<boolean> => {
    while (unsaved.length > 0) {
      const u = unsaved[0];
      if (!(await persist(u.record, u.opts))) return false;
    }
    return true;
  };

  const pauseChain = (reason: PauseReason): void => {
    clearTimeout(flashTimer);
    goTo({ kind: 'resume' });
    openPause(reason);
  };

  /** auto mode: the trial is already over, the next one may be running: a failed save pauses. */
  const onSaveFailed = (): void => {
    if (disposed || finishing) return;
    if (recorder.isRunning()) {
      recorder.abort('save_failed'); // its result opens the pause sheet
      return;
    }
    clearTimeout(flashTimer);
    if (st.view.kind === 'recording') goTo({ kind: 'resume' });
    openPause('save_failed');
  };

  /** Saves without waiting (the chain goes on); auto mode pauses when the save fails. */
  const saveInBackground = (record: TrialRecord, opts?: SaveOptions): void => {
    noteSaved(record);
    void persist(record, opts).then((saved) => {
      if (!saved && autoMode()) onSaveFailed();
    });
  };

  const finish = async (): Promise<void> => {
    const cur = getState().session;
    if (!cur || finishing) return;
    finishing = true;
    clearTimeout(flashTimer);
    // a trial still running ends here (kept if it reached GO, like any stopped attempt)
    if (recorder.isRunning()) recorder.abort('unmount');
    set({ busy: true });
    await retryUnsaved(); // one more chance for a failed save; finishing never waits for it to succeed
    try {
      // completed when every trial of the non-optional blocks is done, else ended_early
      await finishSession(getState().session ?? cur, endStatusOf);
    } catch (e) {
      console.error('the session could not be finished', e);
      showToast(strings().session.saveFailed, 'error');
      finishing = false;
      set({ busy: false });
      return;
    }
    releaseWakeLock();
    if (!disposed) navigate('done');
  };

  const startRun = (planned: PlannedTrial, o: { chained: boolean; gripHint: boolean; awaitReturn: boolean; speechMs?: number }): void => {
    const cur = getState().session;
    if (!cur || disposed || finishing) return;
    clearTimeout(flashTimer);
    const c = planned.condition;
    const auto = autoMode();
    // reading time: a new instruction (compared with the previous trial of the chain) and the words said
    let readMs = auto ? readTimeMs(o.chained ? lastRun : null, c) : 0;
    if (o.speechMs) readMs = Math.max(readMs, o.speechMs);
    // said while the participant gets still; outside a tap, so never on iOS
    if (o.chained && auto && perTrialSpeechAllowed(cur.device)) readMs = Math.max(readMs, speechMs(speak(trialWords(c, strings()))));
    const block = blockOf(cur, planned.block_index);
    instructed = isChoiceTrial(c)
      ? { hand: null, orientation: null, free: true }
      : { hand: block?.hand ?? c.hand, orientation: block?.orientation ?? c.orientation, free: false };
    lastRun = c;
    const info: RunInfo = { planned, attempt: nextAttempt(planned.trial_index), chained: o.chained, gripHint: o.gripHint, flash: null };
    run = info;
    goTo({ kind: 'recording', run: info });
    recorder.start(cur, planned, info.attempt, {
      readyCue: !o.chained,
      awaitReturn: o.awaitReturn,
      gripGate: auto,
      idleMs: auto ? IDLE_MS : null,
      readMs,
    });
  };

  /** Where the session goes from the cursor (fromBlock: the block of the trial that just ended, if any). */
  const advance = (fromBlock: number | null, awaitReturn: boolean): void => {
    const cur = getState().session;
    if (!cur) return;
    const step = nextStep(cur, st.cursor, fromBlock, st.bonus);
    if (step === 'finished') {
      void finish();
    } else if (step === 'checkpoint') {
      chainBlock = null;
      goTo({ kind: 'checkpoint' });
    } else if (step === 'intro') {
      chainBlock = null;
      showIntro();
    } else {
      const planned = plannedAt(cur, st.cursor);
      if (planned) startRun(planned, { chained: true, gripHint: false, awaitReturn });
    }
  };

  /** Session-level warnings of a record that reached GO (see the module doc). */
  const annotate = (record: TrialRecord | null): TrialRecord | null => {
    if (!record || !reachedGo(record) || record.simulated || wakeLockHeld()) return record;
    if (record.quality.warnings.includes(WAKE_LOCK_NOT_HELD)) return record;
    return { ...record, quality: { ...record.quality, warnings: [...record.quality.warnings, WAKE_LOCK_NOT_HELD] } };
  };

  const finishedAuto = (record: TrialRecord | null, uiAbort: UiAbort | null): void => {
    const current = run;
    const status = record?.outcome.status;
    // the next tap screen arms late when the phone may still be at the face
    st.slowArm = armLate(record);
    if (record && status === 'complete') {
      // complete, also when an abort (page hidden, back) cut the return of the arm short: the movement
      // itself is recorded (the engine adds the warning 'return_cut_short')
      failStreak = 0;
      if (isChoiceTrial(record.condition)) {
        const placeholder = awaitingHandReport(record);
        if (uiAbort === 'unmount') {
          noteSaved(placeholder);
          void persist(placeholder);
          return;
        }
        saveInBackground(placeholder);
        goTo({ kind: 'hand', record });
        return;
      }
      const decided = autoDecision(record);
      if (uiAbort === 'unmount') {
        noteSaved(decided);
        void persist(decided);
        return;
      }
      saveInBackground(decided);
      if (decided.outcome.accepted) set({ cursor: record.trial_index + 1 });
      if (uiAbort || !current) {
        pauseChain(pauseReasonOf(uiAbort));
        return;
      }
      const awaitReturn = returnMissed(record);
      if (!decided.outcome.accepted) {
        // held in another grip: the same planned trial again, with the grip picture
        startRun(current.planned, { chained: true, gripHint: true, awaitReturn });
        return;
      }
      advance(record.block_index, awaitReturn);
      return;
    }
    if (uiAbort === 'unmount') {
      if (reachedGo(record)) {
        const failed = asFailedAttempt(record, uiAbortWarning(uiAbort));
        noteSaved(failed);
        void persist(failed);
      }
      return;
    }
    if (!current || !record || uiAbort || (status !== 'no_movement' && status !== 'too_long')) {
      // stopped (pause, back, hidden, stall, idle, failed save, an engine error): kept only if it reached GO
      if (reachedGo(record)) saveInBackground(asFailedAttempt(record, uiAbortWarning(uiAbort)));
      const gripStuck = uiAbort === 'idle' && !!record?.quality.warnings.includes(WRONG_GRIP_AT_STOP);
      pauseChain(gripStuck ? 'wrong_grip' : pauseReasonOf(uiAbort));
      return;
    }
    // no_movement / too_long: kept (not accepted), the reason shows, the same trial runs again
    saveInBackground(asFailedAttempt(record, null));
    failStreak += 1;
    if (failStreak >= 2) {
      failStreak = 0;
      pauseChain('failed_twice');
      return;
    }
    goTo({ kind: 'recording', run: { ...current, flash: status } });
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => {
      if (disposed || st.pause || st.view.kind !== 'recording') return;
      startRun(current.planned, { chained: true, gripHint: false, awaitReturn: false });
    }, RETRY_FLASH_MS);
  };

  const finishedTap = (record: TrialRecord | null, uiAbort: UiAbort | null): void => {
    if (record && record.outcome.status === 'complete') {
      settled.add(record.trial_id);
      if (isChoiceTrial(record.condition)) {
        saveInBackground(awaitingHandReport(record));
        goTo({ kind: 'hand', record });
        return;
      }
      // saved before the review shows (also when the screen went away meanwhile): no event that ends
      // the page reliably finishes an IndexedDB write, so the review must not hold the only copy
      noteSaved(asUnreviewed(record));
      void persist(asUnreviewed(record));
      goTo({ kind: 'review', record });
      return;
    }
    const reason = failureKind(record, uiAbort);
    if (!record) {
      goTo({ kind: 'failed', reason, saving: false });
      return;
    }
    goTo({ kind: 'failed', reason, saving: true });
    const failed = asFailedAttempt(record, uiAbortWarning(uiAbort));
    noteSaved(failed);
    void persist(failed).then(() => {
      // the save may resolve before the failed view has even rendered
      const v = st.view;
      if (v.kind === 'failed') goTo({ ...v, saving: false });
    });
  };

  /** tap mode, the review left unanswered (page hidden, screen gone): saved as not accepted. */
  const protectReview = (): void => {
    const v = st.view;
    if (v.kind === 'review' && !settled.has(v.record.trial_id)) {
      settled.add(v.record.trial_id);
      void persist(asUnreviewed(v.record));
    }
  };

  /** Starts the first trial after a tap (block intro, Ready), asking for the iOS permission first. */
  const startAfterPermission = (
    planned: PlannedTrial,
    permission: Promise<PermissionState> | null,
    denied: () => void,
    said: number,
  ): void => {
    const begin = (): void => {
      // back was pressed while the permission prompt was open: the pause sheet comes first
      if (st.pause || disposed) return;
      startRun(planned, { chained: false, gripHint: false, awaitReturn: false, speechMs: said });
    };
    if (!permission) {
      begin();
      return;
    }
    permission.then(
      (p) => {
        const capsNow = getState().capabilities;
        if (capsNow) setState({ capabilities: { ...capsNow, permission_state: p } });
        if (p === 'denied') denied();
        else begin();
      },
      () => begin(),
    );
  };

  /** How long GO waits for words said in a tap: the speech, plus iOS' ducking of other audio after it. */
  const speechWait = (text: string | null): number => {
    if (!text) return 0;
    const device = getState().session?.device ?? getState().device;
    const webkit = device?.platform === 'ios' || device?.engine === 'webkit';
    return speechMs(text) + (webkit ? SPEECH_DUCK_TAIL_MS : 0);
  };

  /** Reads the stored attempts (attempt numbers, the last movement); retried when the read fails. */
  const readAttempts = (cur: SessionRecord, tries: number): void => {
    const route = (s: SessionRecord): void => {
      set({ cursor: s.progress.next_trial_index });
      const step = nextStep(s, s.progress.next_trial_index, null, false);
      instructed = null; // nothing was done on this page yet: the intro names the hand plainly
      if (step === 'finished') void finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else showIntro();
    };
    attemptSummary(cur).then(
      (sum) => {
        if (disposed) return;
        for (const [i, n] of sum.counts) attempts.counts.set(i, Math.max(n, attempts.counts.get(i) ?? 0));
        for (const id of sum.ids) attempts.ids.add(id);
        const moved = sum.latestMoved;
        if (!lastMoved && moved) {
          lastMoved = { trial_id: moved.trial_id, trial_index: moved.trial_index, accepted: moved.outcome.accepted, condition: moved.condition };
        }
        // saves still running when the screen appeared are done now (the summary waited for them):
        // nothing could start before this, so continue from the freshest progress
        const fresh = getState().session;
        if (
          fresh &&
          fresh.session_id === cur.session_id &&
          fresh.status === 'in_progress' &&
          fresh.progress.next_trial_index !== st.cursor &&
          !finishing &&
          st.pause === null &&
          (st.view.kind === 'intro' || st.view.kind === 'checkpoint')
        ) {
          route(fresh);
        }
        set({ attemptsReady: true, canRedo: redoAllowed(), redoTarget: lastMoved });
      },
      (e) => {
        console.error('the stored attempts could not be read', e);
        if (disposed) return;
        if (tries + 1 < ATTEMPT_READ_TRIES) {
          // the intro's Start stays disabled meanwhile: an attempt number must never repeat
          summaryTimer = setTimeout(() => readAttempts(getState().session ?? cur, tries + 1), ATTEMPT_READ_RETRY_MS);
          return;
        }
        attemptFloor = ATTEMPT_FLOOR; // numbers above every stored attempt: unique without the read
        set({ attemptsReady: true });
      },
    );
  };

  return {
    state: () => ({ ...st }),
    nextAttempt,

    enter() {
      const cur = getState().session;
      if (!cur) return;
      if (cur.status !== 'in_progress') {
        navigate('done');
        return;
      }
      set({ cursor: cur.progress.next_trial_index });
      const step = nextStep(cur, cur.progress.next_trial_index, null, false);
      if (step === 'finished') void finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else showIntro();
      readAttempts(cur, 0);
    },

    dispose() {
      if (disposed) return;
      clearTimeout(flashTimer);
      clearTimeout(summaryTimer);
      protectReview();
      disposed = true;
      releaseWakeLock();
    },

    finished(record, uiAbort, error) {
      if (error) console.warn('trial ended without a normal result', error);
      const rec = annotate(record);
      if (autoMode()) finishedAuto(rec, uiAbort);
      else finishedTap(rec, uiAbort);
    },

    blockStart() {
      if (st.busy || !st.attemptsReady || finishing || recorder.isRunning()) return;
      const cur = getState().session;
      const planned = cur ? plannedAt(cur, st.cursor) : null;
      if (!cur || !planned) return;
      const block = blockOf(cur, planned.block_index);
      st.slowArm = false;
      if (!autoMode()) {
        unlockAudio();
        const w = strings();
        if (!isChoiceTrial(planned.condition)) {
          speak([w.common.hands[block?.hand ?? planned.condition.hand], w.session.speech.grips[block?.orientation ?? planned.condition.orientation]]);
        }
        goTo({ kind: 'ready' });
        return;
      }
      // iOS: the permission prompt must be requested synchronously inside the tap, before any await
      const permission = requestPermissionInTap();
      unlockAudio();
      acquireWakeLock();
      const said = speak(introWords(block, planned, strings()));
      chainBlock = planned.block_index;
      failStreak = 0;
      startAfterPermission(planned, permission, () => showIntro({ notice: 'permission_denied' }), speechWait(said));
    },

    ready() {
      if (st.busy || !st.attemptsReady || finishing || recorder.isRunning()) return;
      const cur = getState().session;
      const planned = cur ? plannedAt(cur, st.cursor) : null;
      if (!cur || !planned) return;
      const permission = requestPermissionInTap();
      unlockAudio(); // iOS: audio must also be unlocked inside the tap
      acquireWakeLock();
      const w = strings();
      speak([isChoiceTrial(planned.condition) ? null : w.common.hands[planned.condition.hand], ...trialWords(planned.condition, w)]);
      startAfterPermission(planned, permission, () => goTo({ kind: 'ready', notice: 'permission_denied' }), 0);
    },

    async decide(decision) {
      const v = st.view;
      // two taps in the same frame must not both save
      if (v.kind !== 'review' || deciding) return;
      deciding = true;
      const record = applyDecision(v.record, decision);
      settled.add(record.trial_id);
      set({ busy: true });
      noteSaved(record);
      const saved = await persist(record);
      deciding = false;
      set({ busy: false });
      if (!saved) {
        settled.delete(record.trial_id);
        return;
      }
      if (decision.kind !== 'accept') {
        goTo({ kind: 'ready' });
        return;
      }
      set({ cursor: saved.progress.next_trial_index });
      const step = nextStep(saved, st.cursor, record.block_index, st.bonus);
      if (step === 'finished') await finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else goTo(step === 'intro' ? { kind: 'break', fromBlock: record.block_index } : { kind: 'ready' });
    },

    retryFailed() {
      if (st.view.kind !== 'failed') return;
      unlockAudio();
      goTo({ kind: 'ready' });
    },

    nextBlock() {
      if (st.view.kind !== 'break') return;
      unlockAudio();
      showIntro();
    },

    async answerHand(answer) {
      const v = st.view;
      if (v.kind !== 'hand' || deciding) return;
      deciding = true;
      st.slowArm = false;
      set({ busy: true });
      const record = withReportedHand(v.record, answer);
      noteSaved(record);
      // "not sure": never re-run, the plan moves on past it
      const saved = await persist(record, answer === 'unsure' ? { skip: true } : undefined);
      deciding = false;
      set({ busy: false });
      if (!saved) return; // the toast says why; the question stays
      // the hand the free trial was done with: the first intro words itself against it
      instructed = { hand: answer === 'unsure' ? null : answer, orientation: null, free: true };
      chainBlock = null;
      set({ cursor: record.trial_index + 1 });
      const step = nextStep(saved, st.cursor, null, st.bonus);
      if (step === 'finished') await finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else showIntro();
    },

    async continueSession() {
      if (st.busy || finishing) return;
      unlockAudio();
      acquireWakeLock();
      st.slowArm = false;
      if (unsaved.length > 0) {
        set({ busy: true });
        const ok = await retryUnsaved();
        set({ busy: false });
        if (!ok) return; // still paused: the toast said why
      }
      set({ pause: null });
      if (st.view.kind !== 'resume') return; // back to the screen under the sheet
      const cur = getState().session;
      if (!cur) return;
      failStreak = 0;
      // inside the block whose intro was tapped the trial runs at once; anywhere else, its intro
      const step = nextStep(cur, st.cursor, chainBlock, st.bonus);
      const planned = plannedAt(cur, st.cursor);
      if (step === 'trial' && planned) startRun(planned, { chained: false, gripHint: false, awaitReturn: false });
      else if (step === 'finished') void finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else showIntro();
    },

    async revoke(flag) {
      const target = lastMoved;
      const cur = getState().session;
      if (st.busy || finishing || !cur || !target || !redoAllowed()) return;
      unlockAudio();
      acquireWakeLock();
      st.slowArm = false;
      set({ busy: true });
      let updated: SessionRecord;
      try {
        if (!(await retryUnsaved())) throw new Error('an earlier attempt is still unsaved');
        if (target.accepted) {
          updated = await revokeAccepted(getState().session ?? cur, target.trial_id, flag);
        } else {
          // not accepted: the same planned trial runs again anyway; "wrong hand" is only noted on it
          updated = flag ? await flagAttempt(getState().session ?? cur, target.trial_id, flag) : (getState().session ?? cur);
        }
      } catch (e) {
        console.error('redo failed', e);
        showToast(strings().session.saveFailed, 'error');
        set({ busy: false });
        return;
      }
      if (lastMoved?.trial_id === target.trial_id) lastMoved = null;
      chainBlock = null; // the intro names the hand again
      failStreak = 0;
      clearTimeout(flashTimer);
      set({ busy: false, canRedo: redoAllowed(), redoTarget: lastMoved, cursor: updated.progress.next_trial_index, pause: null });
      const step = nextStep(updated, st.cursor, null, st.bonus);
      if (step === 'finished') void finish();
      else if (step === 'checkpoint') goTo({ kind: 'checkpoint' });
      else showIntro({ redo: true });
    },

    async finishNow() {
      if (st.busy) return;
      await finish();
    },

    async later() {
      if (st.busy) return;
      if (unsaved.length > 0) {
        set({ busy: true });
        await retryUnsaved();
        set({ busy: false });
      }
      releaseWakeLock();
      showToast(strings().session.pause.laterToast, 'info');
      navigate('welcome');
    },

    startBonus() {
      if (st.busy || finishing || st.view.kind !== 'checkpoint') return;
      unlockAudio();
      acquireWakeLock();
      st.slowArm = false;
      set({ bonus: true });
      showIntro();
    },

    setSitting(sitting) {
      const cur = getState().session;
      if (!cur) return;
      setPosture(cur, sitting ? 'sitting' : 'standing').catch((e) => {
        console.error('the posture could not be saved', e);
        showToast(strings().session.saveFailed, 'error');
      });
    },

    openPause(reason = 'paused') {
      st.slowArm = false;
      openPause(reason);
    },

    back() {
      const cur = getState().session;
      if (!cur || cur.status !== 'in_progress') return false;
      if (recorder.isRunning()) recorder.abort('user');
      else if (st.view.kind === 'recording') pauseChain('paused'); // the retry flash
      openPause(st.pause ?? 'paused');
      return true;
    },

    pageHidden() {
      if (recorder.isRunning()) recorder.abort('hidden');
      else if (autoMode() && st.view.kind === 'recording') pauseChain('hidden'); // the retry flash
      protectReview();
    },
  };
}
