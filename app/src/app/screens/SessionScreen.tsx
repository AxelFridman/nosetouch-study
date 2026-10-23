/**
 * Participant session runner: block intro → trial instruction (Ready) → live recording → review →
 * (next trial | block break → block intro) → … → finishSession('completed') → Done screen.
 *
 * Everything is derived from state.session (progress.next_trial_index), so a resumed session continues
 * at the right trial; the local view state only says which step of the current trial is on screen.
 *  - Ready tap: the iOS motion-permission request (when still needed) and the audio unlock run
 *    synchronously inside the tap; the attempt number is fetched BEFORE the tap.
 *  - The live panel re-renders at most ~15 Hz (immediately on phase/message changes), never per sample.
 *  - Touches anywhere on the panel go to controller.addTouch. The only control is a long-press Stop,
 *    disabled while the phone travels to / rests at the nose (where the nose could press it).
 *  - Page hidden → abort('interrupted'); no new motion sample for 2 s → abort('interrupted') and a
 *    sensors-stalled message. The UI reason is appended to quality.warnings ('ui_abort:…').
 *  - Failed attempts are saved automatically (accepted=false). A complete attempt is saved the moment
 *    it ends, BEFORE the review shows, as not accepted (outcome.note 'review_interrupted'): a reload,
 *    a back gesture or a closed tab during the review can no longer lose it (IndexedDB work started
 *    in pagehide does not finish). The decision (Next / Redo / Problem) re-saves the same trial_id,
 *    which saveAttempt counts as a change of acceptance, not as a new attempt.
 *  - System back button / edge swipe (setBackHandler): never leaves the session. It stops a running
 *    trial like the Stop control (reason 'user') and opens the Pause dialog; the guard history entry
 *    is restored, so repeated presses are caught too.
 *  - Taps within TAP_GUARD_MS of a step change are ignored: Start, Ready, Try again, Continue and the
 *    review buttons share screen positions with the step before, so a double tap must not press the
 *    next step's button.
 */
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { JSX } from 'preact';
import type {
  DeviceInfo,
  ParticipantFlag,
  PermissionState,
  PhoneRotation,
  PlannedTrial,
  SessionRecord,
  TrialLiveState,
  TrialMessage,
  TrialPhase,
  TrialRecord,
} from '../../core/types';
import { ORIENTATIONS } from '../../core/constants';
import { createTrialController, type TrialController } from '../../core/trialController';
import { getScreenOrientation, requestMotionPermission } from '../../core/sensors';
import { Button, Card, Notice, ProgressBar, Screen } from '../../ui/components';
import { PhoneFigure } from '../../ui/PhoneFigure';
import { finiteMax, motionMagnitudes, SignalPlot, type SignalMarker, type SignalSeries } from '../../ui/SignalPlot';
import { stringsFor, useStrings, type Strings } from '../../i18n';
import { getServices } from '../services';
import { createTapGuard } from '../state';
import { getState, navigate, setBackHandler, setState, showToast, useAppState } from '../store';
import { deniedContext } from './SensorCheckScreen';
import {
  applyDecision,
  asFailedAttempt,
  asUnreviewed,
  blockOf,
  blockPosition,
  buildTrialOptions,
  finishSession,
  nextAttemptNumber,
  nextPlannedTrial,
  saveAttempt,
  stepAfterAccept,
  type TrialDecision,
} from '../sessionFlow';
import '../../ui/session.css';

/** Watchdog: abort when no motion sample has arrived for this long. */
export const STALL_MS = 2000;
/** Live panel refresh period (≈ 15 Hz). */
export const RENDER_MS = 66;
/** If the controller does not deliver a record this long after an abort, give up on it. */
const FINISH_FALLBACK_MS = 3000;
const FLAGS: ParticipantFlag[] = ['wrong_hand', 'wrong_orientation', 'other_problem'];

// ───────────────────────────── pure helpers (exported for tests) ─────────────────────────────

export type UiAbort = 'user' | 'hidden' | 'stalled' | 'unmount' | 'start_failed';
export type FailureKind = 'no_movement' | 'too_long' | 'aborted' | 'interrupted' | 'sensor_stalled' | 'error';
export type PanelTone = 'wait' | 'go' | 'move' | 'done' | 'error';

const ERROR_MESSAGES: TrialMessage[] = ['too_early', 'no_movement', 'too_long', 'sensor_stalled'];

/** Background colour family of the live panel for a phase/message. */
export function panelTone(phase: TrialPhase | null, message: TrialMessage | null): PanelTone {
  if (message && ERROR_MESSAGES.includes(message)) return 'error';
  switch (phase) {
    case 'armed':
      return 'go';
    case 'moving':
      return 'move';
    case 'post_roll':
    case 'done':
      return 'done';
    default:
      return 'wait';
  }
}

/** Why an attempt failed, for the participant-facing message. */
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

/** The Stop control works before the movement only: at the nose, the nose itself could long-press it. */
export function stopAllowed(phase: TrialPhase | null): boolean {
  return phase === null || phase === 'waiting_still' || phase === 'foreperiod' || phase === 'armed';
}

/** The Ready screen's advice after a denied motion permission, for the browser the page runs in. */
export function permissionDeniedText(t: Strings['session']['ready']['permissionDenied'], device: DeviceInfo | null): string {
  const ctx = deniedContext(device);
  switch (ctx.kind) {
    case 'ios_safari':
      return t.safari;
    case 'ios_home_screen':
      return t.homeScreen;
    case 'ios_browser':
      return t.browser(ctx.browser);
    case 'ios_in_app':
      return t.inApp;
    case 'android':
      return t.android;
    default:
      return t.other;
  }
}

function safeScreenOrientation(): { type: string | null; angle: number | null } {
  try {
    return getScreenOrientation();
  } catch {
    return { type: null, angle: null };
  }
}

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

/**
 * Speaks `parts` when speech is enabled. MUST be called synchronously inside a tap handler: iOS
 * Safari only speaks while a user gesture is being processed (timers/effects stay silent).
 */
function speak(parts: (string | null | false | undefined)[]): void {
  try {
    const { cues } = getServices();
    const { settings } = getState();
    if (!settings.speech || !cues.say) return;
    const text = parts.filter(Boolean).join('. ');
    if (text) cues.say(text, settings.language);
  } catch {
    // speech is a convenience; the screen and the beeps carry the protocol
  }
}

/** What is said at the Ready tap: useful when the screen will face away from the participant. */
function trialSpeech(c: PlannedTrial['condition']): string[] {
  const words = stringsFor(getState().settings.language).common;
  return [
    words.hands[c.hand],
    words.speeds[c.speed].label,
    c.start_position !== 'natural' ? words.startPositions[c.start_position].label : '',
    c.variant !== 'standard' ? words.variants[c.variant].label : '',
  ];
}

/** Saves an attempt into its own session; never throws (a toast reports failures). */
export async function persistAttempt(record: TrialRecord): Promise<SessionRecord | null> {
  try {
    const cur = getState().session;
    const base =
      cur && cur.session_id === record.session_id ? cur : await getServices().store.getSession(record.session_id);
    if (!base) throw new Error(`session ${record.session_id} not found`);
    return await saveAttempt(base, record);
  } catch (e) {
    console.error('saving the attempt failed', e);
    showToast(stringsFor(getState().settings.language).session.saveFailed, 'error');
    return null;
  }
}

/**
 * Saves a complete attempt the moment it ends, before its review shows: not accepted, outcome.note
 * 'review_interrupted'. The decision re-saves the same trial_id later (saveAttempt counts that as a
 * change of acceptance, never as another attempt). Marks it settled, so the protective save on
 * pagehide/unmount has nothing left to do.
 */
export function saveForReview(record: TrialRecord, settled: Set<string>): Promise<SessionRecord | null> {
  settled.add(record.trial_id);
  return persistAttempt(asUnreviewed(record));
}

// ───────────────────────────── recorder hook ─────────────────────────────

type FinishHandler = (record: TrialRecord | null, uiAbort: UiAbort | null, error?: unknown) => void;

interface Run {
  controller: TrialController;
  settle: (record: TrialRecord | null, error?: unknown) => void;
  fallback: ReturnType<typeof setTimeout> | undefined;
}

interface Recorder {
  live: TrialLiveState | null;
  stalled: boolean;
  active: boolean;
  start(session: SessionRecord, planned: PlannedTrial, attempt: number): void;
  abort(reason: UiAbort): void;
  touch(type: 'start' | 'end' | 'cancel', e: TouchEvent): void;
}

/** Owns one TrialController at a time: throttled live state, watchdog, abort reasons. */
function useTrialRecorder(onFinished: FinishHandler): Recorder {
  const [live, setLive] = useState<TrialLiveState | null>(null);
  const [stalled, setStalled] = useState(false);
  const [active, setActive] = useState(false);
  const runRef = useRef<Run | null>(null);
  const abortRef = useRef<UiAbort | null>(null);
  const mounted = useRef(true);
  const handler = useRef(onFinished);
  handler.current = onFinished;

  const abort = (reason: UiAbort) => {
    const run = runRef.current;
    if (!run) return;
    if (!abortRef.current) abortRef.current = reason;
    try {
      run.controller.abort(reason === 'user' ? 'user' : 'interrupted');
    } catch (e) {
      console.warn('trial abort failed', e);
    }
    // a controller that never resolves must not leave the participant stuck on the panel
    if (!run.fallback) {
      run.fallback = setTimeout(() => run.settle(null, new Error('the trial did not finish after abort')), FINISH_FALLBACK_MS);
    }
  };

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      abort('unmount');
    };
  }, []);

  const start = (session: SessionRecord, planned: PlannedTrial, attempt: number) => {
    if (runRef.current) return;
    abortRef.current = null;
    const buf = { live: null as TrialLiveState | null, seq: 0, shown: 0, key: '' };
    let ticker: ReturnType<typeof setInterval> | undefined;
    let settled = false;
    const run: Run = {
      controller: null as unknown as TrialController,
      fallback: undefined,
      settle: (record, error) => {
        if (settled) return;
        settled = true;
        clearInterval(ticker);
        clearTimeout(run.fallback);
        if (runRef.current === run) runRef.current = null;
        if (mounted.current) setActive(false);
        handler.current(record, abortRef.current, error);
      },
    };

    try {
      const services = getServices();
      const source = services.createSource({ condition: planned.condition });
      run.controller = createTrialController(
        buildTrialOptions(session, planned, attempt, {
          source,
          cues: services.cues,
          onLive: (ls) => {
            buf.live = ls;
            buf.seq++;
            const key = `${ls.phase}|${ls.message ?? ''}`;
            if (key !== buf.key) {
              // phase changes (GO!) are shown at once; everything else waits for the ticker
              buf.key = key;
              buf.shown = buf.seq;
              if (mounted.current) setLive({ ...ls });
            }
          },
          getScreenOrientation: safeScreenOrientation,
          isHidden: () => typeof document !== 'undefined' && document.hidden,
        }),
      );
    } catch (e) {
      run.settle(null, e);
      return;
    }

    runRef.current = run;
    setLive(null);
    setStalled(false);
    setActive(true);

    const controller = run.controller;
    let lastSeen = controller.lastSampleAt;
    let lastChange = performance.now();
    ticker = setInterval(() => {
      if (buf.seq !== buf.shown) {
        buf.shown = buf.seq;
        if (mounted.current && buf.live) setLive({ ...buf.live });
      }
      const seen = controller.lastSampleAt;
      const now = performance.now();
      if (seen !== lastSeen) {
        lastSeen = seen;
        lastChange = now;
      } else if (now - lastChange > STALL_MS && !abortRef.current) {
        if (mounted.current) setStalled(true);
        abort('stalled');
      }
    }, RENDER_MS);

    controller.result.then(
      (record) => run.settle(record),
      (err) => run.settle(null, err),
    );
    controller.start().catch((err) => {
      console.error('trial start failed', err);
      if (!abortRef.current) abortRef.current = 'start_failed';
      abort('start_failed');
    });
  };

  const touch = (type: 'start' | 'end' | 'cancel', e: TouchEvent) => {
    const run = runRef.current;
    if (!run) return;
    const first = e.changedTouches[0] ?? e.touches[0];
    try {
      run.controller.addTouch(type, e.touches.length, first ? first.clientX : null, first ? first.clientY : null);
    } catch {
      // touches are auxiliary data; never break the trial for them
    }
  };

  return { live, stalled, active, start, abort, touch };
}

// ───────────────────────────── small components ─────────────────────────────

function ProgressRing(props: { value: number }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, props.value));
  return (
    <svg class="tp-ring" viewBox="0 0 132 132" aria-hidden="true">
      <circle class="tp-ring-track" cx="66" cy="66" r={r} />
      <circle
        class="tp-ring-value"
        cx="66"
        cy="66"
        r={r}
        stroke-dasharray={c.toFixed(2)}
        stroke-dashoffset={((1 - v) * c).toFixed(2)}
        transform="rotate(-90 66 66)"
      />
    </svg>
  );
}

/** Stop control that only fires after a ~800 ms press (a brushing nose or cheek must not stop a trial). */
function HoldToStop(props: { disabled: boolean; onStop: () => void; label: string; aria: string }) {
  const [pressing, setPressing] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const cancel = () => {
    clearTimeout(timer.current);
    timer.current = undefined;
    setPressing(false);
  };
  const begin = () => {
    if (props.disabled || timer.current) return;
    setPressing(true);
    timer.current = setTimeout(() => {
      timer.current = undefined;
      setPressing(false);
      props.onStop();
    }, 800);
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (props.disabled) cancel();
  }, [props.disabled]);
  return (
    <button
      type="button"
      class={`tp-stop ${pressing ? 'is-pressing' : ''}`}
      disabled={props.disabled}
      aria-label={props.aria}
      data-testid="btn-stop-hold"
      onPointerDown={begin}
      onPointerUp={cancel}
      onPointerLeave={cancel}
      onPointerCancel={cancel}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !e.repeat) {
          e.preventDefault();
          begin();
        }
      }}
      onKeyUp={cancel}
      onContextMenu={(e) => e.preventDefault()}
    >
      <span class="tp-stop-fill" aria-hidden="true" />
      <span class="tp-stop-text">{props.label}</span>
    </button>
  );
}

function TrialPanel(props: { planned: PlannedTrial; recorder: Recorder; simulated: boolean }) {
  const s = useStrings();
  const { live, stalled } = props.recorder;
  const c = props.planned.condition;
  const phase = live?.phase ?? null;
  const message: TrialMessage | null = stalled ? 'sensor_stalled' : (live?.message ?? null);
  const tone = panelTone(phase, message);
  const big = message ? s.session.live.messages[message] : live ? s.session.live.messages.hold_still : s.session.live.starting;
  const holding = !stalled && (phase === null || phase === 'waiting_still' || phase === 'foreperiod') && message !== 'too_early';
  const progress = phase === 'foreperiod' ? 1 : (live?.still_progress ?? 0);
  const onTouch = (type: 'start' | 'end' | 'cancel') => (e: JSX.TargetedTouchEvent<HTMLDivElement>) =>
    props.recorder.touch(type, e);

  return (
    <div
      class="trial-panel"
      data-testid="trial-panel"
      data-phase={phase ?? 'waiting_still'}
      data-tone={tone}
      onTouchStart={onTouch('start')}
      onTouchEnd={onTouch('end')}
      onTouchCancel={onTouch('cancel')}
    >
      <div class="tp-top">
        <span class="tp-chip">{s.common.handShort[c.hand]}</span>
        <span class="tp-condition" data-testid="tp-condition">
          {s.common.speeds[c.speed].label}
          {c.start_position !== 'natural' ? ` · ${s.common.startPositions[c.start_position].label}` : ''}
          {c.variant !== 'standard' ? ` · ${s.common.variants[c.variant].label}` : ''}
        </span>
        {props.simulated ? <span class="tp-chip">{s.session.live.sim}</span> : <span aria-hidden="true" />}
      </div>
      {holding && <ProgressRing value={progress} />}
      <p class="tp-message" aria-live="assertive">
        {big}
      </p>
      {holding && live && <p class="tp-sub">{s.session.live.waitBeep}</p>}
      <HoldToStop
        disabled={stalled || !stopAllowed(phase)}
        onStop={() => props.recorder.abort('user')}
        label={s.session.live.stop}
        aria={s.session.live.stopAria}
      />
    </div>
  );
}

function InstructionRow(props: { label: string; value: string; how?: string; special?: boolean; testId?: string }) {
  return (
    <div class={`instruction-row ${props.special ? 'is-special' : ''}`} data-testid={props.testId}>
      <span class="instruction-label">{props.label}</span>
      <span class="instruction-value">{props.value}</span>
      {props.how && <span class="instruction-how">{props.how}</span>}
    </div>
  );
}

function rotationText(s: Strings, r: PhoneRotation | 'unknown' | null): string {
  return r && r !== 'unknown' ? s.session.rotations[r] : '?';
}

function TrialReview(props: { record: TrialRecord; busy: boolean; onDecide: (d: TrialDecision) => void }) {
  const s = useStrings();
  const [flagOpen, setFlagOpen] = useState(false);
  // The review takes the place of the live panel, and the flag menu that of the buttons: a tap
  // meant for what was there a moment ago (a double tap on "Problem") must not decide.
  const [tap] = useState(createTapGuard);
  tap.shown(flagOpen ? 'flags' : 'review');
  const decide = (d: TrialDecision) => {
    if (!tap.tooSoon()) props.onDecide(d);
  };
  const r = props.record;
  const sig = useMemo(
    () =>
      motionMagnitudes(r.streams.motion, {
        smoothingMs: r.segmentation.params.smoothing_ms,
        gravityTauMs: r.segmentation.params.gravity_tau_ms,
      }),
    [r],
  );
  const series = useMemo((): SignalSeries[] => {
    const gPeak = finiteMax(sig.gyro) ?? 0;
    const aPeak = finiteMax(sig.acc) ?? 0;
    // both traces normalised to their own peak: the plot shows timing, the legend the magnitudes
    const norm = (v: number[], peak: number) => (peak > 0 ? v.map((x) => x / peak) : v);
    return [
      { values: norm(sig.gyro, gPeak), color: 'var(--primary)', label: s.session.review.rotation(gPeak.toFixed(0)) },
      { values: norm(sig.acc, aPeak), color: 'var(--warn)', label: s.session.review.acceleration(aPeak.toFixed(1)) },
    ];
  }, [sig, s]);
  const markers = useMemo((): SignalMarker[] => {
    const m = s.session.review.markers;
    const out: SignalMarker[] = [];
    if (r.timing.go_ms !== null) out.push({ t: r.timing.go_ms, label: m.go, color: 'var(--phase-go)' });
    if (r.timing.onset_ms !== null) out.push({ t: r.timing.onset_ms, label: m.onset, color: 'var(--hand-left)' });
    if (r.timing.end_ms !== null) out.push({ t: r.timing.end_ms, label: m.end, color: 'var(--phase-done)' });
    if (r.timing.first_touch_ms !== null) out.push({ t: r.timing.first_touch_ms, label: m.touch, color: 'var(--muted)' });
    return out;
  }, [r, s]);

  const duration = r.segmentation.duration_ms;
  const expected = ORIENTATIONS[r.condition.orientation]?.rotation ?? null;
  const hz = r.quality.motion_hz;

  return (
    <section class="stack" data-testid="trial-review">
      <div class="review-head">
        <h2>{s.session.review.heading}</h2>
        <span class="review-duration">
          {duration !== null ? s.session.review.duration((duration / 1000).toFixed(2)) : s.session.review.noDuration}
        </span>
      </div>
      <SignalPlot
        t={sig.t}
        series={series}
        markers={markers}
        height={150}
        yMin={0}
        yMax={1.08}
        yTicks={false}
        ariaLabel={s.session.review.plotAria}
        testId="review-plot"
      />
      {r.detected.orientation_match === false && (
        <Notice kind="warn" testId="orientation-warning">
          {s.session.review.orientationMismatch(rotationText(s, expected), rotationText(s, r.detected.rotation_start))}
        </Notice>
      )}
      <p class="muted small">{s.session.review.details(r.quality.n_motion, hz !== null ? hz.toFixed(0) : '–')}</p>
      <p class="big">
        <strong>{s.session.review.question}</strong>
      </p>
      {flagOpen ? (
        <div class="flag-menu" data-testid="flag-menu">
          <p>
            <strong>{s.session.review.flagTitle}</strong>
          </p>
          {FLAGS.map((f) => (
            <Button
              key={f}
              variant="secondary"
              block
              disabled={props.busy}
              testId={`btn-flag-${f}`}
              onClick={() => decide({ kind: 'flag', flag: f })}
            >
              {s.session.review.flags[f]}
            </Button>
          ))}
          <Button variant="ghost" block disabled={props.busy} onClick={() => setFlagOpen(false)}>
            {s.session.review.flagCancel}
          </Button>
        </div>
      ) : (
        <div class="stack-sm">
          <Button size="xl" block disabled={props.busy} testId="btn-next" onClick={() => decide({ kind: 'accept' })}>
            {props.busy ? s.session.saving : s.session.review.next}
          </Button>
          <div class="review-actions">
            <Button variant="secondary" disabled={props.busy} testId="btn-redo" onClick={() => decide({ kind: 'redo' })}>
              {s.session.review.redo}
            </Button>
            <Button variant="secondary" disabled={props.busy} testId="btn-problem" onClick={() => setFlagOpen(true)}>
              {s.session.review.problem}
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}

// ───────────────────────────── screen ─────────────────────────────

type View =
  | { kind: 'intro' }
  | { kind: 'ready'; notice?: 'permission_denied' }
  | { kind: 'recording' }
  | { kind: 'review'; record: TrialRecord }
  | { kind: 'failed'; reason: FailureKind; saving: boolean }
  | { kind: 'break'; fromBlock: number };

export function SessionScreen() {
  const st = useAppState();
  const s = useStrings();
  const session = st.session;
  const [view, setView] = useState<View>({ kind: 'intro' });
  const [busy, setBusy] = useState(false);
  const [exitOpen, setExitOpen] = useState(false);
  const [attempt, setAttempt] = useState<{ trialIndex: number; nAttempts: number; n: number } | null>(null);
  const mounted = useRef(true);
  const viewRef = useRef(view);
  viewRef.current = view;
  // trial_ids that must not be protectively saved again (already saved, or a decision is under way)
  const settledIds = useRef(new Set<string>());
  const deciding = useRef(false);
  // read by callbacks that outlive a render (the permission prompt, the back button)
  const exitOpenRef = useRef(exitOpen);
  exitOpenRef.current = exitOpen;
  const openExit = () => {
    exitOpenRef.current = true;
    setExitOpen(true);
  };
  const closeExit = () => {
    exitOpenRef.current = false;
    setExitOpen(false);
  };
  // double-tap guard: taps within TAP_GUARD_MS of a step change are ignored (see the module doc)
  const [tap] = useState(createTapGuard);
  tap.shown(`${view.kind}|${exitOpen}`);

  const next = session ? nextPlannedTrial(session) : null;
  const total = session?.protocol.n_planned_trials ?? 0;
  const done = session?.progress.next_trial_index ?? 0;

  const goTo = (v: View) => {
    if (mounted.current) setView(v);
  };

  // Each step starts at the top: the Start/Next buttons sit at the bottom of long screens, and a
  // participant must not miss the hand / orientation instruction at the top of the next one.
  useEffect(() => {
    try {
      window.scrollTo(0, 0);
    } catch {
      // not available (tests)
    }
  }, [view.kind]);

  const finish = async (status: 'completed' | 'abandoned') => {
    const cur = getState().session;
    if (!cur) return;
    setBusy(true);
    try {
      await finishSession(cur, status);
    } catch (e) {
      console.error(e);
      showToast(s.session.saveFailed, 'error');
      if (mounted.current) setBusy(false);
      return;
    }
    releaseWakeLock();
    navigate('done');
  };

  const onFinished: FinishHandler = (record, uiAbort, error) => {
    if (error) console.warn('trial ended without a normal result', error);
    if (record && record.outcome.status === 'complete') {
      // saved before the review shows (also when the screen went away meanwhile): no event that
      // ends the page reliably finishes an IndexedDB write, so the review must not hold the only copy
      void saveForReview(record, settledIds.current);
      goTo({ kind: 'review', record });
      return;
    }
    const reason = failureKind(record, uiAbort);
    if (!record) {
      goTo({ kind: 'failed', reason, saving: false });
      return;
    }
    goTo({ kind: 'failed', reason, saving: true });
    void persistAttempt(asFailedAttempt(record, uiAbortWarning(uiAbort))).then(() => {
      // functional update: the save may resolve before the failed view has even rendered
      if (mounted.current) setView((v) => (v.kind === 'failed' ? { ...v, saving: false } : v));
    });
  };

  const recorder = useTrialRecorder(onFinished);
  const recorderRef = useRef(recorder);
  recorderRef.current = recorder;

  // System back button / edge swipe: stay. A running trial stops as if Stop had been held (kept as an
  // 'aborted' attempt), then the Pause dialog offers to pause, end or go on. Returning true makes
  // the store push its guard entry again, so the next press lands here too.
  useEffect(
    () =>
      setBackHandler(() => {
        const cur = getState().session;
        if (!cur || cur.status !== 'in_progress') return false;
        recorderRef.current.abort('user');
        openExit();
        return true;
      }),
    [],
  );

  // protective save of an unanswered review + abort of a running trial when the page is hidden
  useEffect(() => {
    const protect = () => {
      recorderRef.current.abort('hidden');
      const v = viewRef.current;
      if (v.kind === 'review' && !settledIds.current.has(v.record.trial_id)) {
        settledIds.current.add(v.record.trial_id);
        void persistAttempt(asUnreviewed(v.record));
      }
    };
    const onVisibility = () => {
      if (document.hidden) protect();
    };
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pagehide', protect);
    mounted.current = true;
    return () => {
      mounted.current = false;
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('pagehide', protect);
      const v = viewRef.current;
      if (v.kind === 'review' && !settledIds.current.has(v.record.trial_id)) {
        settledIds.current.add(v.record.trial_id);
        void persistAttempt(asUnreviewed(v.record));
      }
      releaseWakeLock();
    };
  }, []);

  // on entry: a finished session goes to Done; an in-progress one with no trial left is completed
  useEffect(() => {
    const cur = getState().session;
    if (!cur) return;
    if (cur.status !== 'in_progress') navigate('done');
    else if (!nextPlannedTrial(cur)) void finish('completed');
  }, []);

  // the attempt number is needed synchronously in the Ready tap, so fetch it beforehand; it can only
  // change when an attempt is saved (n_attempts) or the planned trial changes
  const nAttempts = session?.progress.n_attempts ?? 0;
  useEffect(() => {
    if (!session || !next) return;
    let alive = true;
    const key = { trialIndex: next.trial_index, nAttempts };
    nextAttemptNumber(session, next.trial_index).then(
      (n) => alive && setAttempt({ ...key, n }),
      (e) => {
        console.error(e);
        if (alive) setAttempt({ ...key, n: 1 });
      },
    );
    return () => {
      alive = false;
    };
  }, [session?.session_id, next?.trial_index, nAttempts]);
  const attemptOk = !!attempt && !!next && attempt.trialIndex === next.trial_index && attempt.nAttempts === nAttempts;

  // block intro: block-start chime (Web Audio works outside gestures once unlocked). The spoken
  // hand + orientation is said in the Start tap instead — iOS ignores speech outside gestures.
  const introBlock = view.kind === 'intro' && next ? next.block_index : null;
  useEffect(() => {
    if (introBlock === null || !session) return;
    if (!blockOf(session, introBlock)) return;
    try {
      getServices().cues.play('block_start');
    } catch {
      // cues are best effort
    }
  }, [introBlock, session?.session_id]);

  const onReady = () => {
    if (tap.tooSoon() || !session || !next || !attemptOk || !attempt || busy || recorder.active) return;
    const st0 = getState();
    const caps = st0.capabilities;
    // iOS: the permission prompt must be requested synchronously inside the tap, before any await
    let permission: Promise<PermissionState> | null = null;
    if (!st0.settings.simulate && caps?.permission_api && caps.permission_state !== 'granted') {
      try {
        permission = requestMotionPermission();
      } catch (e) {
        console.warn('motion permission request failed', e);
      }
    }
    unlockAudio(); // iOS: audio must also be unlocked inside the tap
    acquireWakeLock();
    speak(trialSpeech(next.condition));
    const plannedNow = next;
    const attemptNow = attempt.n;
    const sessionNow = session;
    setView({ kind: 'recording' });
    const begin = () => {
      // back was pressed while the permission prompt was open: the Pause dialog comes first
      if (exitOpenRef.current) goTo({ kind: 'ready' });
      else recorder.start(sessionNow, plannedNow, attemptNow);
    };
    if (!permission) {
      begin();
      return;
    }
    permission.then(
      (p) => {
        const capsNow = getState().capabilities;
        if (capsNow) setState({ capabilities: { ...capsNow, permission_state: p } });
        if (p === 'denied') goTo({ kind: 'ready', notice: 'permission_denied' });
        else begin();
      },
      () => begin(),
    );
  };

  const decide = async (decision: TrialDecision) => {
    const v = viewRef.current;
    // ref guard: two taps in the same frame must not both save
    if (v.kind !== 'review' || deciding.current) return;
    deciding.current = true;
    const record = v.record;
    settledIds.current.add(record.trial_id);
    setBusy(true);
    const saved = await persistAttempt(applyDecision(record, decision));
    deciding.current = false;
    if (mounted.current) setBusy(false);
    if (!saved) {
      settledIds.current.delete(record.trial_id);
      return;
    }
    if (decision.kind !== 'accept') {
      goTo({ kind: 'ready' });
      return;
    }
    const step = stepAfterAccept(saved, record.block_index);
    if (step === 'finished') await finish('completed');
    else goTo(step === 'block_break' ? { kind: 'break', fromBlock: record.block_index } : { kind: 'ready' });
  };

  const pauseAndExit = () => {
    releaseWakeLock();
    showToast(s.session.exit.paused, 'info');
    navigate('welcome');
  };

  // ── render ──

  if (!session) {
    return (
      <Screen title={s.common.appName} testId="screen-session">
        <Notice kind="info">{s.session.noSession}</Notice>
        <Button block onClick={() => navigate('welcome')} testId="btn-home">
          {s.session.goHome}
        </Button>
      </Screen>
    );
  }

  if (view.kind === 'recording' && next) {
    return (
      <Screen className="session-screen is-recording" testId="screen-session">
        <TrialPanel planned={next} recorder={recorder} simulated={st.settings.simulate} />
      </Screen>
    );
  }

  // The header's Pause button is not offered during the review (it asks for a decision); the system
  // back button opens the same dialog from every step, the review included (its attempt is saved).
  const canExit = view.kind !== 'review' && !busy;
  const headerRight = canExit ? (
    <Button variant="ghost" testId="btn-session-exit" onClick={openExit}>
      {s.session.exit.open}
    </Button>
  ) : undefined;

  if (exitOpen && !busy) {
    return (
      <Screen title={s.session.exit.title} headerRight={undefined} testId="screen-session">
        <Card testId="session-exit">
          <p>{s.session.exit.body}</p>
          <div class="exit-actions">
            <Button
              block
              testId="btn-exit-pause"
              onClick={() => {
                if (!tap.tooSoon()) pauseAndExit();
              }}
            >
              {s.session.exit.pause}
            </Button>
            <Button
              block
              variant="danger"
              testId="btn-exit-end"
              onClick={() => {
                if (!tap.tooSoon()) void finish('abandoned');
              }}
            >
              {s.session.exit.end}
            </Button>
            <Button block variant="ghost" testId="btn-exit-cancel" onClick={closeExit}>
              {s.session.exit.cancel}
            </Button>
          </div>
        </Card>
      </Screen>
    );
  }

  if (!next) {
    // all planned trials accepted: finishing is under way (see the entry effect / decide)
    return (
      <Screen title={s.common.appName} testId="screen-session">
        <p class="muted">{s.session.saving}</p>
      </Screen>
    );
  }

  const block = blockOf(session, next.block_index);
  const hand = block?.hand ?? next.condition.hand;
  const orientation = block?.orientation ?? next.condition.orientation;
  const pos = blockPosition(session, next.block_index);
  const trialTitle = s.session.trialOf(next.trial_index + 1, total);
  const blockTitle = s.session.blockOf(pos.position, pos.count);
  const progressRight = s.session.trialsDone(done, total);

  if (view.kind === 'break') {
    const from = blockPosition(session, view.fromBlock);
    const cheers = s.session.breakScreen.cheers;
    return (
      <Screen
        title={s.session.blockOf(from.position, from.count)}
        headerRight={headerRight}
        testId="screen-session"
        footer={
          <>
            <Button
              size="xl"
              block
              testId="btn-next-block"
              onClick={() => {
                if (tap.tooSoon()) return;
                unlockAudio();
                setView({ kind: 'intro' });
              }}
            >
              {s.session.breakScreen.continue}
            </Button>
            <Button
              variant="secondary"
              block
              testId="btn-pause-exit"
              onClick={() => {
                if (!tap.tooSoon()) pauseAndExit();
              }}
            >
              {s.session.breakScreen.pauseExit}
            </Button>
          </>
        }
      >
        <section class="stack session-center" data-testid="block-break">
          <p class="break-cheer">{cheers[(from.position - 1) % cheers.length]}</p>
          <h2>{s.session.breakScreen.heading(from.position, from.count)}</h2>
          <div style={{ width: '100%' }}>
            <ProgressBar value={total > 0 ? done / total : 0} label={progressRight} />
          </div>
          <p>{s.session.breakScreen.rest}</p>
          <p>
            <strong>{s.session.breakScreen.next(s.common.hands[hand], s.common.orientations[orientation].label)}</strong>
          </p>
        </section>
      </Screen>
    );
  }

  if (view.kind === 'intro') {
    const prev = pos.position > 1 ? session.protocol.plan[pos.position - 2] : null;
    const newHand = !!prev && prev.hand !== hand;
    const needsSafety = block?.kind === 'odd' || !!block?.trials.some((t) => t.condition.speed === 'fast');
    return (
      <Screen
        title={blockTitle}
        headerRight={headerRight}
        testId="screen-session"
        footer={
          <Button
            size="xl"
            block
            testId="btn-block-start"
            onClick={() => {
              if (tap.tooSoon()) return;
              unlockAudio();
              speak([s.common.hands[hand], s.common.orientations[orientation].label]);
              setView({ kind: 'ready' });
            }}
          >
            {s.session.intro.start}
          </Button>
        }
      >
        {/* the notices come right under the hand, above the figure: below it they sat under the
            sticky Start button on common phone screens */}
        <section class="stack session-center" data-testid="block-intro">
          <div style={{ width: '100%' }}>
            <ProgressBar value={(pos.position - 1) / pos.count} label={blockTitle} right={progressRight} />
          </div>
          {newHand && <p class="new-hand">{s.session.intro.newHand}</p>}
          <span class={`hand-badge hand-badge-lg hand-${hand}`}>{s.common.hands[hand]}</span>
          {block?.kind === 'odd' && (
            <Notice kind="info" testId="intro-odd">
              {s.session.intro.odd}
            </Notice>
          )}
          {block?.kind === 'free' && (
            <Notice kind="info" testId="intro-free">
              {s.session.intro.free}
            </Notice>
          )}
          {needsSafety ? (
            <Notice kind="warn" testId="intro-safety">
              {s.common.safety}
            </Notice>
          ) : (
            <p class="muted small" data-testid="intro-safety">
              {s.common.safety}
            </p>
          )}
          <div class="intro-figure">
            <PhoneFigure orientation={orientation} hand={hand} size={220} />
          </div>
          <p class="session-hold">{s.common.holdWith(hand)}</p>
          <h2>{s.common.orientations[orientation].label}</h2>
          <p class="session-sub">{s.common.orientations[orientation].how}</p>
          <p>{s.common.movementSummary}</p>
          {session.simulated !== st.settings.simulate && <Notice kind="info">{s.session.simMismatch}</Notice>}
        </section>
      </Screen>
    );
  }

  if (view.kind === 'review') {
    return (
      <Screen title={trialTitle} testId="screen-session">
        <TrialReview record={view.record} busy={busy} onDecide={(d) => void decide(d)} />
      </Screen>
    );
  }

  if (view.kind === 'failed') {
    return (
      <Screen
        title={trialTitle}
        headerRight={headerRight}
        testId="screen-session"
        footer={
          <Button
            size="xl"
            block
            disabled={view.saving}
            testId="btn-retry"
            onClick={() => {
              if (tap.tooSoon()) return;
              unlockAudio();
              setView({ kind: 'ready' });
            }}
          >
            {view.saving ? s.session.saving : s.session.failed.retry}
          </Button>
        }
      >
        <section class="stack" data-testid="trial-failed">
          <Card tone="warn" title={s.session.failed.heading}>
            <p>{s.session.failed.reasons[view.reason]}</p>
            <p class="muted small">{s.session.failed.kept}</p>
          </Card>
        </section>
      </Screen>
    );
  }

  // ready: the instruction of the next trial. The four rows come first and are compact, so that on
  // common phone screens (360×640, 390×664) all of them stay above the sticky Ready button; the
  // figure follows, small, and only on tall screens (session.css). A denial notice goes on top: with
  // it the trial cannot start anyway.
  const c = next.condition;
  const trialsInBlock = block?.trials.length ?? 1;
  return (
    <Screen
      title={trialTitle}
      headerRight={headerRight}
      className="session-ready"
      testId="screen-session"
      footer={
        <Button size="xl" block disabled={!attemptOk || busy} testId="btn-ready" onClick={onReady}>
          {attemptOk ? s.session.ready.ready : s.session.ready.preparing}
        </Button>
      }
    >
      <section class="stack trial-ready" data-testid="trial-ready">
        <ProgressBar
          value={total > 0 ? done / total : 0}
          label={s.session.trialInBlock(next.trial_in_block + 1, trialsInBlock)}
          right={progressRight}
        />
        <div class="instruction-head">
          <span class={`hand-badge hand-${c.hand}`}>{s.common.hands[c.hand]}</span>
          {attempt && attemptOk && attempt.n > 1 && <span class="badge">{s.session.attempt(attempt.n)}</span>}
        </div>
        {view.kind === 'ready' && view.notice === 'permission_denied' && (
          <Notice kind="error" testId="ready-permission-denied">
            {permissionDeniedText(s.session.ready.permissionDenied, st.device)}
          </Notice>
        )}
        <div class="instruction">
          <InstructionRow testId="instr-phone" label={s.session.ready.phone} value={s.common.orientations[c.orientation].label} />
          <InstructionRow
            testId="instr-speed"
            label={s.session.ready.speed}
            value={s.common.speeds[c.speed].label}
            how={s.common.speeds[c.speed].how}
            special={c.speed !== 'natural'}
          />
          <InstructionRow
            testId="instr-start"
            label={s.session.ready.startPosition}
            value={s.common.startPositions[c.start_position].label}
            how={s.common.startPositions[c.start_position].how}
            special={c.start_position !== 'natural'}
          />
          <InstructionRow
            testId="instr-movement"
            label={s.session.ready.movement}
            value={s.common.variants[c.variant].label}
            how={s.common.variants[c.variant].how}
            special={c.variant !== 'standard'}
          />
        </div>
        <div class="ready-figure" data-testid="ready-figure">
          <PhoneFigure orientation={c.orientation} size={96} />
        </div>
        <p class="muted">{s.session.ready.hint}</p>
      </section>
    </Screen>
  );
}
