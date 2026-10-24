/**
 * Participant session screen. The session's logic (the hands-free chain, the hand question, the pause
 * sheet, the checkpoint, the tap-review mode, saving and finishing) lives in sessionRunner.ts, which
 * is unit-tested; this file renders runner.state() and forwards to it:
 *  - the taps (How-to card btn-try / block intro btn-block-start, hand-report buttons, pause sheet,
 *    checkpoint, and the tap mode's Ready / review / break / retry);
 *  - the recorder's results (useTrialRecorder: one TrialController at a time, the live panel's state,
 *    the stall watchdog and the no-GO idle limit);
 *  - the system back button (setBackHandler) and page visibility (visibilitychange → hidden, pagehide).
 *
 * Cards (spec v2 §1): How-to card (howto-card: gesture-demo animation, three steps, "I'm sitting"
 * toggle tgl-sitting, btn-try), hand question (hand-report), block intro (block-intro: tier label,
 * "Phone in your LEFT hand" / "Keep the phone in your LEFT hand", POV animation + grip text, what
 * varies, btn-block-start naming the hand), live panel (trial-panel with data-phase and
 * data-trial-index: huge hand and speed words, the variant / start-position instruction, a status line
 * by phase, the grip picture on a wrong grip, progress bar, "about N min left", btn-pause: hold ≈800 ms),
 * pause sheet (pause-sheet inside session-exit), checkpoint (checkpoint: btn-finish, btn-bonus).
 *
 * UI rules:
 *  - The live panel re-renders at most ~15 Hz (immediately on phase/message changes), never per sample.
 *    It is read at arm's length: the hand and the speed are the biggest words; a special movement or
 *    start position takes the speed word's place with one short how-line; "Read this first…" shows
 *    while GO waits for the reading time; a wrong grip is said in words (the grip picture only shows
 *    when the page turned with the phone: gripFigureFits).
 *  - Touches anywhere on the panel go to controller.addTouch with their contact details.
 *  - No motion sample for STALL_MS → abort('stalled'); auto mode: no GO within IDLE_MS → abort('idle').
 *  - Taps within TAP_GUARD_MS of a step change are ignored, and the buttons of a screen that appears
 *    without a tap of the participant (after a trial: the phone may still be near the face) stay
 *    disabled for ARM_DELAY_MS — ARM_DELAY_SLOW_MS after a trial whose arm was not seen going back
 *    out (runner.slowArm) — so a nose or a cheek cannot press them. The pause sheet's Finish now takes
 *    two taps; its Exit for now and the header's Pause go through the same guards.
 *  - Nothing animates while a trial records: the animations are on the intro screens only.
 */
import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import type { ComponentChildren, JSX } from 'preact';
import type {
  DeviceInfo,
  Hand,
  ParticipantFlag,
  PhoneRotation,
  PlannedBlock,
  PlannedTrial,
  SessionRecord,
  TrialCondition,
  TrialLiveState,
  TrialMessage,
  TrialPhase,
  TrialRecord,
} from '../../core/types';
import { ORIENTATIONS } from '../../core/constants';
import { createTrialController, type TrialController } from '../../core/trialController';
import { getScreenOrientation, readTouch } from '../../core/sensors';
import { Button, Card, Notice, ProgressBar, Screen, Toggle } from '../../ui/components';
import { GestureAnimation, type GestureContact, type GestureVariant } from '../../ui/GestureAnimation';
import { PhoneFigure } from '../../ui/PhoneFigure';
import { finiteMax, motionMagnitudes, SignalPlot, type SignalMarker, type SignalSeries } from '../../ui/SignalPlot';
import { useStrings, type Strings } from '../../i18n';
import { getServices } from '../services';
import { createTapGuard, TAP_GUARD_MS } from '../state';
import { navigate, setBackHandler, useAppState } from '../store';
import { deniedContext } from './SensorCheckScreen';
import {
  blockChange,
  blockOf,
  blockPosition,
  blockSpeeds,
  buildTrialOptions,
  introWording,
  isChoiceTrial,
  plannedAt,
  planIntroRef,
  progressOf,
  reviewModeOf,
  roundMinutes,
  secondsLeft,
  type IntroRef,
  type TrialDecision,
} from '../sessionFlow';
import {
  createSessionRunner,
  type PauseReason,
  type RecorderPort,
  type RecorderStart,
  type RedoTarget,
  type RunInfo,
  type SessionRunner,
  type UiAbort,
} from '../sessionRunner';
import { GOOD_MOTION_HZ, MIN_MOTION_HZ } from '../sensorCheck';
import '../../ui/session.css';

// the session logic's helpers, also reachable from here (tests, older imports)
export {
  failureKind,
  IDLE_MS,
  introWords,
  pauseReasonOf,
  perTrialSpeechAllowed,
  persistAttempt,
  reachedGo,
  RETRY_FLASH_MS,
  saveForReview,
  trialWords,
  uiAbortWarning,
  type FailureKind,
  type PauseReason,
  type UiAbort,
} from '../sessionRunner';

/** Watchdog: abort when no motion sample has arrived for this long. */
export const STALL_MS = 2000;
/** Live panel refresh period (≈ 15 Hz). */
export const RENDER_MS = 66;
/** Buttons of a screen that appeared without a tap stay disabled this long (see the module doc). */
export const ARM_DELAY_MS = 900;
/**
 * ... and this long when the trial before it did not see the arm go back out (or ran too long): the
 * phone may still be at the face, where a nose or a cheek could press the next screen's buttons.
 */
export const ARM_DELAY_SLOW_MS = 2500;
/** "Hold to pause" released before it fired: "Keep holding…" shows this long. */
export const HOLD_HINT_MS = 1500;
/** If the controller does not deliver a record this long after an abort, give up on it. */
const FINISH_FALLBACK_MS = 3000;
const FLAGS: ParticipantFlag[] = ['wrong_hand', 'wrong_orientation', 'other_problem'];

// ───────────────────────────── pure helpers (exported for tests) ─────────────────────────────

export type PanelTone = 'wait' | 'go' | 'move' | 'done' | 'error';
/** A key of the live panel's status line texts (strings.session.live.status). */
export type StatusKey = keyof Strings['session']['live']['status'];

const ERROR_MESSAGES: TrialMessage[] = ['too_early', 'no_movement', 'too_long', 'sensor_stalled', 'wrong_grip'];

/** Background colour family of the live panel for a phase/message. */
export function panelTone(phase: TrialPhase | null, message: TrialMessage | null): PanelTone {
  if (message && ERROR_MESSAGES.includes(message)) return 'error';
  switch (phase) {
    case 'armed':
      return 'go';
    case 'moving':
      return 'move';
    case 'post_roll':
    case 'returning':
    case 'done':
      return 'done';
    default:
      return 'wait';
  }
}

/** The live panel's status line: a special message first, else by phase. */
export function statusKey(phase: TrialPhase | null, message: TrialMessage | null): StatusKey {
  switch (message) {
    case 'too_early':
    case 'wrong_grip':
    case 'sensor_stalled':
    case 'no_movement':
    case 'too_long':
    case 'read':
      return message;
    case 'return_arm':
      return 'returning';
    default:
      break;
  }
  switch (phase) {
    case 'waiting_still':
      return 'waiting';
    case 'foreperiod':
      return 'foreperiod';
    case 'armed':
      return 'go';
    case 'moving':
      return 'moving';
    case 'post_roll':
      return 'at_nose';
    case 'returning':
      return 'returning';
    case 'done':
      return 'done';
    default:
      return 'starting';
  }
}

/** The status line of a trial: statusKey, but "Elbow bent — hold still" for the bent_start start pose. */
export function trialStatusKey(phase: TrialPhase | null, message: TrialMessage | null, c: TrialCondition): StatusKey {
  const key = statusKey(phase, message);
  return key === 'waiting' && c.variant === 'bent_start' ? 'waitingBent' : key;
}

/**
 * The live panel's big words: the hand, and in the speed slot the speed — or, for a special movement
 * or start position at natural speed (every variety cell), that movement's short label (NORMAL never
 * changes in the variety blocks). `label` is then shown there and `how` below it.
 */
export function panelWords(c: TrialCondition, s: Strings): { speed: string; special: boolean; label: string | null; how: string | null } {
  const sp =
    c.variant !== 'standard' ? s.common.variants[c.variant] : c.start_position !== 'natural' ? s.common.startPositions[c.start_position] : null;
  if (!sp) return { speed: s.session.live.speedWords[c.speed], special: false, label: null, how: null };
  if (c.speed === 'natural') return { speed: sp.label, special: true, label: null, how: sp.how };
  return { speed: s.session.live.speedWords[c.speed], special: false, label: sp.label, how: sp.how };
}

/**
 * Can the grip picture be trusted? It is drawn in page coordinates, so it is right only when the page
 * turned with the phone: a sideways phone on a landscape page, an upright one on a portrait page. With
 * the rotation locked (common) a sideways phone keeps a portrait page and the picture looks turned by
 * 90° — then only the words show. Unknown readings never show it.
 */
export function gripFigureFits(seen: PhoneRotation | 'unknown' | null | undefined, screenType: string | null): boolean {
  if (!seen || seen === 'unknown' || seen === 'flat' || !screenType) return false;
  const landscape = screenType.startsWith('landscape');
  const sideways = seen === 'top_left' || seen === 'top_right';
  return landscape === sideways;
}

/** The pause control works before the movement only: at the nose, the nose itself could long-press it. */
export function stopAllowed(phase: TrialPhase | null): boolean {
  return phase === null || phase === 'waiting_still' || phase === 'foreperiod' || phase === 'armed';
}

/** The advice after a denied motion permission, for the browser the page runs in. */
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

/** The illustration of a trial: tilted_touch → the angled contact; curved / pause → their variant. */
export function gestureFor(c: TrialCondition): { contact: GestureContact; variant: GestureVariant } {
  return {
    contact: c.variant === 'tilted_touch' ? 'angled' : 'straight',
    variant: c.variant === 'curved' || c.variant === 'pause' ? c.variant : 'standard',
  };
}

/** The intro's line on what changes inside a block: its special movements, else changing speeds. */
export function variesLine(block: PlannedBlock, s: Strings): string | null {
  const items: string[] = [];
  for (const { condition: c } of block.trials) {
    const label =
      c.variant !== 'standard'
        ? s.common.variants[c.variant].label
        : c.start_position !== 'natural'
          ? s.common.startPositions[c.start_position].label
          : null;
    if (label && !items.includes(label)) items.push(label);
  }
  if (items.length > 0) return s.session.intro.variesList(items.join(' · '));
  return blockSpeeds(block).length > 1 ? s.session.intro.speedsVary : null;
}

/**
 * The How-to card's "Play a test beep": inside the tap it unlocks (or repairs) the audio, then plays the
 * GO beep once the context runs. Silent on computers like every cue (the cue sink's own gate).
 */
function playTestBeep(): void {
  try {
    const { cues } = getServices();
    void cues
      .unlock()
      .then(() => cues.play('go'))
      .catch(() => undefined);
  } catch {
    // no audio: nothing to test
  }
}

/** The screen wake lock is held (true when it cannot be told, so no false alarm). */
function wakeLockActive(): boolean {
  try {
    return getServices().wakeLock.active !== false;
  } catch {
    return true;
  }
}

function safeScreenOrientation(): { type: string | null; angle: number | null } {
  try {
    return getScreenOrientation();
  } catch {
    return { type: null, angle: null };
  }
}

// ───────────────────────────── recorder hook ─────────────────────────────

type FinishHandler = (record: TrialRecord | null, uiAbort: UiAbort | null, error?: unknown) => void;

interface Run {
  controller: TrialController;
  settle: (record: TrialRecord | null, error?: unknown) => void;
  fallback: ReturnType<typeof setTimeout> | undefined;
}

interface Recorder extends RecorderPort {
  live: TrialLiveState | null;
  stalled: boolean;
  touch(type: 'start' | 'end' | 'cancel', e: TouchEvent): void;
}

/** Owns one TrialController at a time: throttled live state, watchdogs, abort reasons. */
function useTrialRecorder(onFinished: FinishHandler): Recorder {
  const [live, setLive] = useState<TrialLiveState | null>(null);
  const [stalled, setStalled] = useState(false);
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

  const start = (session: SessionRecord, planned: PlannedTrial, attempt: number, o: RecorderStart) => {
    if (runRef.current) return;
    abortRef.current = null;
    const buf = { live: null as TrialLiveState | null, seq: 0, shown: 0, key: '', goSeen: false };
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
        handler.current(record, abortRef.current, error);
      },
    };

    try {
      const services = getServices();
      // awaitReturn: the simulated participant starts this trial at the nose (real sensors ignore it)
      const source = services.createSource({ condition: planned.condition, startAtNose: o.awaitReturn });
      run.controller = createTrialController(
        buildTrialOptions(session, planned, attempt, {
          source,
          cues: services.cues,
          onLive: (ls) => {
            buf.live = ls;
            buf.seq++;
            if (ls.phase !== 'waiting_still' && ls.phase !== 'foreperiod') buf.goSeen = true;
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
          readyCue: o.readyCue,
          awaitReturn: o.awaitReturn,
          gripGate: o.gripGate,
          readMs: o.readMs,
        }),
      );
    } catch (e) {
      run.settle(null, e);
      return;
    }

    runRef.current = run;
    if (mounted.current) {
      setLive(null);
      setStalled(false);
    }

    const controller = run.controller;
    let lastSeen = controller.lastSampleAt;
    const startedAt = performance.now();
    let lastChange = startedAt;
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
        return;
      }
      if (o.idleMs !== null && !buf.goSeen && now - startedAt > o.idleMs && !abortRef.current) abort('idle');
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
    try {
      // position + contact size/force/angle + the position as a fraction of the viewport (schema 1.2)
      const r = readTouch(e.changedTouches[0] ?? e.touches[0]);
      run.controller.addTouch(type, e.touches.length, r.x, r.y, undefined, r.details);
    } catch {
      // touches are auxiliary data; never break the trial for them
    }
  };

  return { live, stalled, isRunning: () => runRef.current !== null, start, abort, touch };
}

/** false for `ms` after `key` changed (see ARM_DELAY_MS). */
function useArmed(key: string, ms: number = ARM_DELAY_MS): boolean {
  const [armedKey, setArmedKey] = useState<string | null>(null);
  useEffect(() => {
    const t = setTimeout(() => setArmedKey(key), ms);
    return () => clearTimeout(t);
  }, [key]);
  return armedKey === key;
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

/**
 * Pause control that only fires after a ~800 ms press (a nose touch lasts 80–240 ms and must never
 * trigger it). Released too early, it says `holdHint` ("Keep holding…") for HOLD_HINT_MS.
 */
function HoldToPause(props: { disabled: boolean; onPause: () => void; label: string; holdHint: string; aria: string }) {
  const [pressing, setPressing] = useState(false);
  const [hint, setHint] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const hintTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const cancel = (early: boolean) => {
    const wasPressing = timer.current !== undefined;
    clearTimeout(timer.current);
    timer.current = undefined;
    setPressing(false);
    if (early && wasPressing) {
      setHint(true);
      clearTimeout(hintTimer.current);
      hintTimer.current = setTimeout(() => setHint(false), HOLD_HINT_MS);
    }
  };
  const begin = () => {
    if (props.disabled || timer.current) return;
    setPressing(true);
    timer.current = setTimeout(() => {
      timer.current = undefined;
      setPressing(false);
      setHint(false);
      props.onPause();
    }, 800);
  };
  useEffect(
    () => () => {
      clearTimeout(timer.current);
      clearTimeout(hintTimer.current);
    },
    [],
  );
  useEffect(() => {
    if (props.disabled) {
      cancel(false);
      setHint(false);
    }
  }, [props.disabled]);
  return (
    <button
      type="button"
      class={`tp-stop ${pressing ? 'is-pressing' : ''}`}
      disabled={props.disabled}
      aria-label={props.aria}
      data-testid="btn-pause"
      data-hint={hint ? 'keep-holding' : undefined}
      onPointerDown={begin}
      onPointerUp={() => cancel(true)}
      onPointerLeave={() => cancel(false)}
      onPointerCancel={() => cancel(false)}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !e.repeat) {
          e.preventDefault();
          begin();
        }
      }}
      onKeyUp={() => cancel(true)}
      onContextMenu={(e) => e.preventDefault()}
    >
      <span class="tp-stop-fill" aria-hidden="true" />
      <span class="tp-stop-text">{hint && !pressing ? props.holdHint : props.label}</span>
    </button>
  );
}

function TrialPanel(props: {
  run: RunInfo;
  recorder: Recorder;
  simulated: boolean;
  progress: { done: number; total: number };
  minutes: number | null;
}) {
  const s = useStrings();
  const { live, stalled } = props.recorder;
  const { run } = props;
  const c = run.planned.condition;
  const flash = run.flash;
  const phase: TrialPhase | null = flash ? 'done' : (live?.phase ?? (run.chained ? 'waiting_still' : null));
  const message: TrialMessage | null = flash ?? (stalled ? 'sensor_stalled' : (live?.message ?? null));
  const tone = panelTone(phase, message);
  const gripWords = s.session.live.gripWords[c.orientation];
  // a wrong grip is said in words: the picture can look turned when the page does not rotate
  const status = message === 'wrong_grip' && !flash ? gripWords : s.session.live.status[trialStatusKey(phase, message, c)];
  const before = phase === null || phase === 'waiting_still' || phase === 'foreperiod';
  const holding =
    !flash && !stalled && before && message !== 'too_early' && message !== 'wrong_grip' && message !== 'return_arm' && message !== 'read';
  const showGrip = !flash && (message === 'wrong_grip' || (run.gripHint && before));
  const figure = showGrip && gripFigureFits(live?.grip_seen, safeScreenOrientation().type);
  const stillProgress = phase === 'foreperiod' ? 1 : (live?.still_progress ?? 0);
  const choice = isChoiceTrial(c);
  const words = panelWords(c, s);
  const pct = props.progress.total > 0 ? Math.min(1, props.progress.done / props.progress.total) : 0;
  const onTouch = (type: 'start' | 'end' | 'cancel') => (e: JSX.TargetedTouchEvent<HTMLDivElement>) =>
    props.recorder.touch(type, e);

  return (
    <div
      class="trial-panel"
      data-testid="trial-panel"
      data-phase={phase ?? 'waiting_still'}
      data-tone={tone}
      data-trial-index={run.planned.trial_index}
      onTouchStart={onTouch('start')}
      onTouchEnd={onTouch('end')}
      onTouchCancel={onTouch('cancel')}
    >
      <div class="tp-top">
        <div class="tp-progress" aria-hidden="true">
          <div style={{ width: `${(pct * 100).toFixed(1)}%` }} />
        </div>
        <div class="tp-top-row">
          <span class="tp-left" data-testid="tp-time-left">
            {props.minutes !== null ? s.session.live.minutesLeft(props.minutes) : ''}
          </span>
          {props.simulated ? <span class="tp-chip">{s.session.live.sim}</span> : <span aria-hidden="true" />}
        </div>
      </div>
      <div class="tp-words" data-testid="tp-condition">
        <span class={`tp-hand tp-hand-${choice ? 'any' : c.hand}`} data-testid="tp-hand">
          {choice ? s.session.live.anyHand : s.common.handShort[c.hand]}
        </span>
        <span class={`tp-speed ${words.special ? 'tp-speed-special' : ''}`} data-testid="tp-speed">
          {words.speed}
        </span>
      </div>
      {words.how && (
        <p class="tp-variant" data-testid="tp-variant">
          {words.label && <strong>{words.label}</strong>} <span>{words.how}</span>
        </p>
      )}
      {showGrip && (message !== 'wrong_grip' || figure) && (
        <div class="tp-grip" data-testid="tp-grip" data-figure={figure ? 'shown' : 'hidden'}>
          {message !== 'wrong_grip' && <p class="tp-grip-words">{gripWords}</p>}
          {figure && <PhoneFigure orientation={c.orientation} size={104} />}
        </div>
      )}
      {holding && <ProgressRing value={stillProgress} />}
      <p class="tp-message" aria-live="assertive" data-testid="tp-status">
        {status}
      </p>
      <HoldToPause
        disabled={!!flash || stalled || !stopAllowed(phase)}
        onPause={() => props.recorder.abort('user')}
        label={s.session.live.pause}
        holdHint={s.session.live.keepHolding}
        aria={s.session.live.pauseAria}
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

/**
 * Notes for the How-to card from the background sensor check: a slow motion rate (still usable) or no
 * gyroscope; null when there is nothing to say (simulation, no measurement, a good rate).
 */
export function sensorNote(caps: SessionRecord['capabilities'] | null | undefined, simulated: boolean, t: Strings['session']['howTo']): string | null {
  const m = caps?.measured;
  if (simulated || !m) return null;
  const hz = m.motion_hz;
  if (typeof hz === 'number' && Number.isFinite(hz) && hz >= MIN_MOTION_HZ && hz < GOOD_MOTION_HZ) return t.slowSensors(Math.round(hz));
  if (typeof hz === 'number' && hz >= MIN_MOTION_HZ && !m.rotation_available) return t.noGyro;
  return null;
}

/** The intro of the choice block: how to do the movement, then one free trial (btn-try). */
function HowToCard(props: {
  session: SessionRecord;
  headerRight: ComponentChildren;
  enabled: boolean;
  notices: ComponentChildren;
  /** from the background sensor check (sensorNote) */
  sensorNote: string | null;
  /** the screen wake lock is not held: the screen may lock itself in the middle of a block */
  wakeLockNote: boolean;
  onStart: () => void;
  onTestBeep: () => void;
  onSitting: (sitting: boolean) => void;
}) {
  const s = useStrings();
  const t = s.session.howTo;
  return (
    <Screen
      title={t.title}
      headerRight={props.headerRight}
      testId="screen-session"
      footer={
        <>
          <p class="small howto-anyhand" data-testid="howto-anyhand">
            {t.anyHand}
          </p>
          <Button size="xl" block testId="btn-try" disabled={!props.enabled} onClick={props.onStart}>
            {t.try}
          </Button>
        </>
      }
    >
      <section class="stack howto" data-testid="howto-card">
        {props.notices}
        <div class="howto-demo" data-testid="gesture-demo">
          <GestureAnimation hand={null} view="both" speed="natural" contact="straight" badge={false} label={t.gestureAria} />
        </div>
        <div class="howto-angled">
          <div class="howto-angled-figure">
            <GestureAnimation hand={null} view="side" contact="angled" compact badge={false} label={t.angledAria} />
          </div>
          <p class="howto-caption">{t.anyPart}</p>
        </div>
        <ol class="howto-steps">
          {t.steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
        <div class="howto-sound" data-testid="howto-sound">
          <p class="howto-sound-text">{t.sound}</p>
          <Button variant="secondary" testId="btn-test-beep" onClick={props.onTestBeep}>
            {t.testBeep}
          </Button>
        </div>
        <p class="small howto-tip" data-testid="howto-tips">
          {t.tips}
        </p>
        <p class="small howto-tip" data-testid="howto-pause-hint">
          {t.pauseHint}
        </p>
        {props.sensorNote && (
          <Notice kind="warn" testId="howto-sensors">
            {props.sensorNote}
          </Notice>
        )}
        {props.wakeLockNote && (
          <Notice kind="warn" testId="howto-wakelock">
            {t.wakeLock}
          </Notice>
        )}
        <p class="small muted" data-testid="intro-safety">
          {t.safety}
        </p>
        <Toggle
          testId="tgl-sitting"
          checked={props.session.protocol.posture === 'sitting'}
          onChange={props.onSitting}
          label={t.sitting}
        />
      </section>
    </Screen>
  );
}

/**
 * The intro of a block: one tap (naming the hand) starts it. What changes comes first and stays above
 * the Start button on small and landscape screens: the hand line worded against what the participant
 * was last told or did (`introRef`, sessionFlow.introWording), then the grip, then the animation.
 */
function BlockIntro(props: {
  session: SessionRecord;
  planned: PlannedTrial;
  auto: boolean;
  /** the runner's reference for the wording (null: the plan's previous block) */
  introRef: IntroRef | null;
  headerRight: ComponentChildren;
  enabled: boolean;
  notices: ComponentChildren;
  progress: { done: number; total: number };
  minutes: number;
  onStart: () => void;
}) {
  const s = useStrings();
  const { session, planned } = props;
  const block = blockOf(session, planned.block_index);
  const hand = block?.hand ?? planned.condition.hand;
  const orientation = block?.orientation ?? planned.condition.orientation;
  const wording = introWording({ hand, orientation }, props.introRef ?? planIntroRef(session, planned.block_index));
  const change = blockChange(session, planned.block_index);
  const pos = blockPosition(session, planned.block_index);
  const title = typeof block?.tier === 'number' ? s.session.intro.tier(block.tier) : s.session.blockOf(pos.position, pos.count);
  const varies = block ? variesLine(block, s) : null;
  const fast = !!block?.trials.some((t) => t.condition.speed === 'fast');
  const hasChoiceBlock = session.protocol.plan.some((b) => b.kind === 'choice');
  const g = gestureFor(planned.condition);
  const grip = s.common.orientations[orientation];
  const line =
    wording.line === 'phoneIn'
      ? s.session.intro.phoneIn(hand)
      : wording.line === 'turnTo'
        ? s.session.intro.turnTo[orientation]
        : s.session.intro.keepIn(hand);
  return (
    <Screen
      title={title}
      headerRight={props.headerRight}
      testId="screen-session"
      footer={
        <Button size="xl" block testId="btn-block-start" disabled={!props.enabled} onClick={props.onStart}>
          {props.auto ? s.session.intro.startIn(hand) : s.session.intro.start}
        </Button>
      }
    >
      <section
        class="stack session-center block-intro"
        data-testid="block-intro"
        data-block-index={planned.block_index}
        data-intro-line={wording.line}
        data-switch={wording.switchHands ? '1' : '0'}
      >
        {props.notices}
        {wording.switchHands && (
          <p class="new-hand" data-testid="intro-switch">
            {s.session.intro.newHand}
          </p>
        )}
        <p class={`intro-hand hand-${hand}`} data-testid="intro-hand">
          {line}
        </p>
        {wording.sameHand && (
          <p class="intro-same" data-testid="intro-same-hand">
            {s.session.intro.sameHand(hand)}
          </p>
        )}
        <h2 class="intro-grip" data-testid="intro-grip">
          {grip.label}
        </h2>
        <p class="session-sub intro-grip-how">{grip.how}</p>
        <div class="intro-gesture" data-testid="intro-gesture">
          <GestureAnimation
            view="pov"
            compact
            hand={hand}
            orientation={orientation}
            speed="natural"
            contact={g.contact}
            variant={g.variant}
            label={s.session.intro.gestureAria(s.common.hands[hand], grip.label)}
          />
        </div>
        {varies && (
          <p class="intro-varies" data-testid="intro-varies">
            {varies}
          </p>
        )}
        {block?.kind === 'free' && (
          <Notice kind="info" testId="intro-free">
            {s.session.intro.free}
          </Notice>
        )}
        {!props.auto && block?.kind === 'odd' && (
          <Notice kind="info" testId="intro-odd">
            {s.session.intro.odd}
          </Notice>
        )}
        {fast && (
          <p class="muted small" data-testid="intro-safety">
            {s.session.intro.fastGentle}
          </p>
        )}
        {change.afterBasics && (
          <p class="muted small intro-basics" data-testid="intro-encourage">
            {s.session.intro.basicsDone}
          </p>
        )}
        <div class="intro-progress">
          <ProgressBar
            value={props.progress.total > 0 ? props.progress.done / props.progress.total : 0}
            label={s.session.trialsDone(props.progress.done, props.progress.total)}
            right={s.session.live.minutesLeft(props.minutes)}
          />
        </div>
        {!hasChoiceBlock && <p>{s.common.movementSummary}</p>}
      </section>
    </Screen>
  );
}

/** After the choice trial: which hand was it? */
function HandReport(props: { enabled: boolean; onAnswer: (answer: Hand | 'unsure') => void }) {
  const s = useStrings();
  const t = s.session.hand;
  return (
    <Screen testId="screen-session">
      <section class="stack hand-report" data-testid="hand-report">
        <h2 class="hand-question">{t.question}</h2>
        {/* physical sides: the left hand's button stays on the left on right-to-left pages too */}
        <div class="hand-buttons" dir="ltr">
          <Button size="xl" block testId="btn-hand-left" disabled={!props.enabled} onClick={() => props.onAnswer('left')}>
            <span class="hand-choice hand-left">{t.left}</span>
          </Button>
          <Button size="xl" block testId="btn-hand-right" disabled={!props.enabled} onClick={() => props.onAnswer('right')}>
            <span class="hand-choice hand-right">{t.right}</span>
          </Button>
        </div>
        <Button variant="ghost" block testId="btn-hand-unsure" disabled={!props.enabled} onClick={() => props.onAnswer('unsure')}>
          {t.unsure}
        </Button>
      </section>
    </Screen>
  );
}

function Checkpoint(props: { done: number; bonusMinutes: number; enabled: boolean; onFinish: () => void; onBonus: () => void }) {
  const s = useStrings();
  const t = s.session.checkpoint;
  return (
    <Screen
      title={t.title}
      testId="screen-session"
      footer={
        <>
          <Button size="xl" block testId="btn-finish" disabled={!props.enabled} onClick={props.onFinish}>
            {t.finish}
          </Button>
          <Button size="lg" variant="secondary" block testId="btn-bonus" disabled={!props.enabled} onClick={props.onBonus}>
            {t.bonus(Math.max(1, props.bonusMinutes))}
          </Button>
        </>
      }
    >
      <section class="stack session-center" data-testid="checkpoint">
        <p class="checkpoint-body">{t.body(props.done)}</p>
        <p class="muted">{t.bonusHint}</p>
      </section>
    </Screen>
  );
}

/** "LEFT hand · FAST" (or the special movement): which movement the sheet's Redo / wrong hand refer to. */
export function movementText(c: TrialCondition, s: Strings): string {
  const words = panelWords(c, s);
  const hand = isChoiceTrial(c) && c.hand_source !== 'reported' ? s.session.live.anyHand : s.common.hands[c.hand];
  return `${hand} · ${words.label ?? words.speed}`;
}

function PauseSheet(props: {
  reason: PauseReason;
  /** the grip of the current trial, in words (reason 'wrong_grip') */
  gripWords: string | null;
  canRedo: boolean;
  redoTarget: RedoTarget | null;
  enabled: boolean;
  busy: boolean;
  onContinue: () => void;
  onRedo: () => void;
  onWrongHand: () => void;
  onFinish: () => void;
  onLater: () => void;
}) {
  const s = useStrings();
  const t = s.session.pause;
  const live = props.enabled && !props.busy;
  // Finish now takes two taps, at least TAP_GUARD_MS apart: a sheet that opened by itself must not end
  // the visit with one (or a double) nose touch
  const [confirmFinish, setConfirmFinish] = useState(false);
  const firstFinishAt = useRef(0);
  const onFinish = () => {
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!confirmFinish) {
      firstFinishAt.current = now;
      setConfirmFinish(true);
    } else if (now - firstFinishAt.current >= TAP_GUARD_MS) {
      props.onFinish();
    }
  };
  const reasonText = props.reason === 'wrong_grip' && props.gripWords ? t.gripReason(props.gripWords) : t.reasons[props.reason];
  return (
    <Screen title={t.title} testId="screen-session">
      {/* session-exit: the pause dialog's id since v1 (e2e); pause-sheet: v2 */}
      <div class="pause-wrap" data-testid="session-exit">
        <Card testId="pause-sheet">
          <p class="pause-reason" data-testid="pause-reason" data-reason={props.reason}>
            {reasonText}
          </p>
          <div class="pause-actions">
            <Button size="xl" block testId="btn-continue-session" disabled={!live} onClick={props.onContinue}>
              {props.busy ? s.session.saving : t.continue}
            </Button>
            <Button variant="secondary" block testId="btn-redo-last" disabled={!live || !props.canRedo} onClick={props.onRedo}>
              {t.redo}
            </Button>
            <Button
              variant="secondary"
              block
              testId="btn-wrong-hand"
              disabled={!live || !props.canRedo}
              onClick={props.onWrongHand}
            >
              {t.wrongHand}
            </Button>
            {props.canRedo && props.redoTarget && (
              <p class="muted small pause-last" data-testid="pause-last-movement">
                {t.lastMovement(movementText(props.redoTarget.condition, s))}
              </p>
            )}
            <Button
              variant="secondary"
              block
              testId="btn-finish-now"
              disabled={!live}
              onClick={onFinish}
            >
              <span data-confirm={confirmFinish ? '1' : '0'}>{confirmFinish ? t.finishConfirm : t.finish}</span>
            </Button>
            <Button variant="ghost" block testId="btn-exit-pause" disabled={!live} onClick={props.onLater}>
              {t.later}
            </Button>
          </div>
          <p class="muted small">{t.kept}</p>
        </Card>
      </div>
    </Screen>
  );
}

// ───────────────────────────── screen ─────────────────────────────

export function SessionScreen() {
  const st = useAppState();
  const s = useStrings();
  const session = st.session;
  const auto = session ? reviewModeOf(session) === 'auto' : true;

  // the runner drives the session; it reaches the recorder through a ref (the hook needs the runner)
  const [, setTick] = useState(0);
  const recorderRef = useRef<Recorder | null>(null);
  const [runner] = useState<SessionRunner>(() =>
    createSessionRunner(
      {
        start: (cur, planned, attempt, o) => recorderRef.current?.start(cur, planned, attempt, o),
        abort: (reason) => recorderRef.current?.abort(reason),
        isRunning: () => recorderRef.current?.isRunning() ?? false,
      },
      () => setTick((n) => n + 1),
    ),
  );
  const recorder = useTrialRecorder((record, uiAbort, error) => runner.finished(record, uiAbort, error));
  recorderRef.current = recorder;
  const { view, pause, cursor, bonus, busy, attemptsReady, canRedo, redoTarget, introRef, slowArm } = runner.state();

  // double-tap guard: taps within TAP_GUARD_MS of a step change are ignored (see the module doc)
  const [tap] = useState(createTapGuard);
  const stepKey = `${view.kind}|${pause ?? ''}|${cursor}`;
  tap.shown(stepKey);
  // after a trial whose arm was not seen going back out, the next screen arms late (runner.slowArm)
  const armed = useArmed(stepKey, slowArm ? ARM_DELAY_SLOW_MS : ARM_DELAY_MS);
  /** a button press, unless it came too soon after the screen changed (or before the arm delay) */
  const press =
    (fn: () => void, needsArm = false) =>
    () => {
      if (tap.tooSoon() || (needsArm && !armed)) return;
      fn();
    };

  // on entry: a finished session goes to Done; otherwise the stored attempts seed the local counters
  useEffect(() => {
    runner.enter();
    return () => runner.dispose();
  }, []);

  // System back button / edge swipe: stay. A running trial stops as if the pause control had been
  // held, then the pause sheet opens. Returning true makes the store push its guard entry again.
  useEffect(() => setBackHandler(() => runner.back()), []);

  // the page is hidden: a running trial stops (auto: the pause sheet waits for the return); tap mode:
  // an unanswered review is saved protectively
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) runner.pageHidden();
    };
    const onPageHide = () => runner.pageHidden();
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pagehide', onPageHide);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('pagehide', onPageHide);
    };
  }, []);

  // a block intro appears: block-start chime (Web Audio works outside gestures once unlocked). The
  // spoken hand + grip is said in the Start tap instead — iOS ignores speech outside gestures.
  const next = session ? plannedAt(session, cursor) : null;
  const introBlock = view.kind === 'intro' && pause === null && next ? next.block_index : null;
  useEffect(() => {
    if (introBlock === null) return;
    try {
      getServices().cues.play('block_start');
    } catch {
      // cues are best effort
    }
  }, [introBlock, session?.session_id]);

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

  if (pause !== null) {
    const current = plannedAt(session, cursor);
    return (
      <PauseSheet
        reason={pause}
        gripWords={current ? s.session.live.gripWords[current.condition.orientation] : null}
        canRedo={canRedo && attemptsReady}
        redoTarget={redoTarget}
        enabled={armed}
        busy={busy}
        onContinue={press(() => void runner.continueSession())}
        onRedo={press(() => void runner.revoke(null))}
        onWrongHand={press(() => void runner.revoke('wrong_hand'))}
        onFinish={press(() => void runner.finishNow())}
        onLater={press(() => void runner.later(), true)}
      />
    );
  }

  const progress = progressOf(session, cursor, bonus);
  const minutes = roundMinutes(secondsLeft(session, cursor, bonus ? 'include' : 'exclude'));

  if (view.kind === 'recording') {
    return (
      <Screen className="session-screen is-recording" testId="screen-session">
        <TrialPanel run={view.run} recorder={recorder} simulated={st.settings.simulate} progress={progress} minutes={minutes} />
      </Screen>
    );
  }

  const headerRight = !busy ? (
    <Button variant="ghost" testId="btn-session-exit" onClick={press(() => runner.openPause('paused'))}>
      {s.session.pause.open}
    </Button>
  ) : undefined;

  if (view.kind === 'hand') {
    return <HandReport enabled={armed && !busy} onAnswer={(a) => press(() => void runner.answerHand(a), true)()} />;
  }

  if (view.kind === 'checkpoint') {
    return (
      <Checkpoint
        done={progressOf(session, cursor, false).done}
        bonusMinutes={roundMinutes(secondsLeft(session, cursor, 'only'))}
        enabled={armed && !busy}
        onFinish={press(() => void runner.finishNow(), true)}
        onBonus={press(() => runner.startBonus(), true)}
      />
    );
  }

  if (view.kind === 'review') {
    return (
      <Screen title={s.session.trialOf(view.record.trial_index + 1, session.protocol.n_planned_trials)} testId="screen-session">
        <TrialReview record={view.record} busy={busy} onDecide={(d) => void runner.decide(d)} />
      </Screen>
    );
  }

  if (!next || view.kind === 'resume') {
    // finishing is under way (entry / last trial), or a chain paused without its sheet
    return (
      <Screen title={s.common.appName} testId="screen-session">
        {view.kind === 'resume' && next ? (
          <Button size="xl" block testId="btn-continue-session" onClick={() => runner.openPause('paused')}>
            {s.session.pause.continue}
          </Button>
        ) : (
          <p class="muted">{s.session.saving}</p>
        )}
      </Screen>
    );
  }

  const total = session.protocol.n_planned_trials;
  const trialTitle = s.session.trialOf(next.trial_index + 1, total);
  const progressRight = s.session.trialsDone(progress.done, progress.total);
  const block = blockOf(session, next.block_index);
  const deniedNotice =
    (view.kind === 'intro' || view.kind === 'ready') && view.notice === 'permission_denied' ? (
      <Notice kind="error" testId="ready-permission-denied">
        {permissionDeniedText(s.session.ready.permissionDenied, st.device)}
      </Notice>
    ) : null;
  const notices = (
    <>
      {view.kind === 'intro' && view.redo && (
        <Notice kind="info" testId="intro-redo">
          {s.session.intro.redo}
        </Notice>
      )}
      {deniedNotice}
      {session.simulated !== st.settings.simulate && <Notice kind="info">{s.session.simMismatch}</Notice>}
    </>
  );

  if (view.kind === 'break') {
    const from = blockPosition(session, view.fromBlock);
    const cheers = s.session.breakScreen.cheers;
    const hand = block?.hand ?? next.condition.hand;
    const orientation = block?.orientation ?? next.condition.orientation;
    return (
      <Screen
        title={s.session.blockOf(from.position, from.count)}
        headerRight={headerRight}
        testId="screen-session"
        footer={
          <>
            <Button size="xl" block testId="btn-next-block" onClick={press(() => runner.nextBlock())}>
              {s.session.breakScreen.continue}
            </Button>
            <Button variant="secondary" block testId="btn-pause-exit" onClick={() => void runner.later()}>
              {s.session.breakScreen.pauseExit}
            </Button>
          </>
        }
      >
        <section class="stack session-center" data-testid="block-break">
          <p class="break-cheer">{cheers[(from.position - 1) % cheers.length]}</p>
          <h2>{s.session.breakScreen.heading(from.position, from.count)}</h2>
          <div style={{ width: '100%' }}>
            <ProgressBar value={progress.total > 0 ? progress.done / progress.total : 0} label={progressRight} />
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
    const enabled = armed && attemptsReady && !busy;
    const onStart = press(() => runner.blockStart(), true);
    if (isChoiceTrial(next.condition) || block?.kind === 'choice') {
      return (
        <HowToCard
          session={session}
          headerRight={headerRight}
          enabled={enabled}
          notices={notices}
          sensorNote={sensorNote(st.capabilities ?? session.capabilities, st.settings.simulate, s.session.howTo)}
          wakeLockNote={!st.settings.simulate && !wakeLockActive()}
          onStart={onStart}
          onTestBeep={press(playTestBeep)}
          onSitting={(sitting) => runner.setSitting(sitting)}
        />
      );
    }
    return (
      <BlockIntro
        session={session}
        planned={next}
        auto={auto}
        introRef={introRef}
        headerRight={headerRight}
        enabled={enabled}
        notices={notices}
        progress={progress}
        minutes={minutes}
        onStart={onStart}
      />
    );
  }

  if (view.kind === 'failed') {
    return (
      <Screen
        title={trialTitle}
        headerRight={headerRight}
        testId="screen-session"
        footer={
          <Button size="xl" block disabled={view.saving} testId="btn-retry" onClick={press(() => runner.retryFailed())}>
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

  // ready (tap mode): the instruction of the next trial. The four rows come first and are compact, so
  // that on common phone screens (360×640, 390×664) all of them stay above the sticky Ready button; the
  // figure follows, small, and only on tall screens (session.css). A denial notice goes on top: with
  // it the trial cannot start anyway.
  const c = next.condition;
  const trialsInBlock = block?.trials.length ?? 1;
  const attemptN = runner.nextAttempt(next.trial_index);
  return (
    <Screen
      title={trialTitle}
      headerRight={headerRight}
      className="session-ready"
      testId="screen-session"
      footer={
        <Button size="xl" block disabled={!attemptsReady || busy} testId="btn-ready" onClick={press(() => runner.ready())}>
          {attemptsReady ? s.session.ready.ready : s.session.ready.preparing}
        </Button>
      }
    >
      <section class="stack trial-ready" data-testid="trial-ready">
        <ProgressBar
          value={progress.total > 0 ? progress.done / progress.total : 0}
          label={s.session.trialInBlock(next.trial_in_block + 1, trialsInBlock)}
          right={progressRight}
        />
        <div class="instruction-head">
          <span class={`hand-badge hand-${c.hand}`}>{isChoiceTrial(c) ? s.session.live.anyHand : s.common.hands[c.hand]}</span>
          {attemptsReady && attemptN > 1 && <span class="badge">{s.session.attempt(attemptN)}</span>}
        </div>
        {deniedNotice}
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
