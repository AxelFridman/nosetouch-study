/**
 * Runs ONE trial: records all sensor streams, drives the OnlineSegmenter, emits cues and builds
 * the TrialRecord. Entirely driven by incoming motion samples (no setTimeout for trial logic), so it
 * is deterministic with a simulated source and runs in Node. The OnlineSegmenter (online_v1, ported
 * to analysis/motionlab/segment.py) decides still / onset / end; everything around the segmented
 * movement (gates before GO, the false-start policy, the return of the arm) is decided HERE, on the
 * same smoothed signals (segmentation.MotionSignals), so the port needs no change.
 *
 * Flow (v2, protocol 2.0.0):
 *  start(): t0 = source.clock.now(); recorder = new StreamRecorder(t0); cue 'ready' unless
 *           opts.readyCue === false (chained trials of the hands-free flow); source.start(listener)
 *           (offline simulated sources run the whole trial inside this call).
 *  each motion sample → recorder.addMotionRel → signals.push → segmenter.push(rel sample) → events:
 *    'still'        → the GATES must pass before GO is scheduled:
 *                       · awaitReturn: a return movement (smoothed |ω| > RETURN_GYRO_DPS or |a_lin| >
 *                         RETURN_ACC_MS2, continuously ≥ RETURN_SUSTAIN_MS) has been seen in this
 *                         trial (message 'return_arm' until then);
 *                       · gripGate (never for 'free'): the mean accg of the last GRIP_WINDOW_MS is not
 *                         confidently another grip (wrongGrip below; message 'wrong_grip').
 *                     While a gate holds and the segmenter stays 'still', the gates are re-checked on
 *                     every sample. Passing → still_ms = that sample's t, foreperiod = foreperiod_min +
 *                     rng()·(max−min); phase 'foreperiod' (message 'hold_still').
 *    sample t ≥ still_ms + foreperiod while segmenter.state==='still' → GO: cues.play('go'),
 *                     source.notifyCue?.('go', abs t), recorder.addCue, segmenter.markGo(rel t); 'armed'
 *    'still_lost' / 'false_start' → cancel pending GO; phase 'waiting_still'. The false-start policy
 *                     (protocol 2.0.0) ignores the segmenter's own split, which uses the (slow:
 *                     lowered) onset floors: a loss of stillness while a GO was PENDING is a COUNTED
 *                     false start (falseStarts++, cue 'error', message 'too_early') once the smoothed
 *                     motion exceeds FALSE_START_GYRO_DPS or FALSE_START_ACC_MS2 — at the loss itself
 *                     or within FALSE_START_WATCH_MS after it (an anticipation that starts gently is
 *                     still a false start); motion that stays at or below both is silent (posture
 *                     sway: no cue, not counted, the hold restarts). A loss while no GO was pending
 *                     (a gate held, e.g. moving while GO was withheld for the return) is always silent.
 *    'onset'        → phase 'moving' (message 'moving')
 *    'end'          → cues.play('end') (the double beep: "arm out again"); notifyCue('end');
 *                     phase 'post_roll' (message 'return_arm')
 *    'no_movement' / 'too_long' → cues.play('error'); finish with that status
 *  After the end (post-roll WITH the return): recording continues at least post_roll_ms after
 *  end_detected_ms (phase 'post_roll'), then (phase 'returning') until the return of the arm was
 *  seen — a return movement (as for awaitReturn, starting after end_detected_ms) followed by
 *  stillness (still_gyro_dps / still_acc_ms2) held still_hold_ms — or until return_cap_ms after
 *  end_detected_ms, whichever comes first → finish('complete'). params.return_cap_ms absent (v1
 *  params) → 0: the trial ends after post_roll_ms exactly as in protocol 1.0.0.
 *    timing.return_onset_ms = first sample of that return movement; timing.return_end_ms = first
 *    sample of the stillness that followed it (the trial stops still_hold_ms later); null when not
 *    observed. returnMissed(record) tells the session whether the next chained trial needs
 *    awaitReturn: true.
 *  Touch events are recorded via addTouch() (with the schema 1.2 contact details); first
 *  touchstart after GO → timing.first_touch_ms.
 *  abort('user') → finish('aborted'); abort('interrupted') (page hidden etc.) → finish('interrupted');
 *  but in phase 'returning' (movement and minimum post-roll already recorded) an abort — or an
 *  offline source running out — ends the trial as 'complete' with warning RETURN_CUT_SHORT.
 *  finish(): source.stop(); build streams; quality via computeQuality (hidden_during from isHidden
 *    polled on every sample); detected.* via rotationFromGravity on baseline gravity (start) and the
 *    mean accg at the nose (end) = the samples from end_ms to end_detected_ms (the hold that
 *    confirmed the end; the recording now goes on through the return), or, for trials without an
 *    end, the last END_GRAVITY_WINDOW_MS before stop; compared with ORIENTATIONS[cond].rotation;
 *    outcome.accepted = (status === 'complete'); resolve `result`.
 *  The trial_id etc. come from opts.ids. created_at = ISO time of finish (source.clock.epochAt).
 *  A watchdog is NOT part of this class: the UI calls abort('interrupted') if no samples arrive.
 *
 * Grip gate (wrongGrip): rotationFromGravity(mean accg of the last GRIP_WINDOW_MS, gravity sign of
 *  opts.calibration, instructed screen facing) — default 35° cone, 'unknown' never blocks. A gravity
 *  sign that no measured upright hold confirms (signVerified: the engine prior, or none) could be
 *  wrong and would turn upright into upside_down and top_left into top_right, so with an unverified
 *  sign a mismatch only blocks when it survives flipping the sign (e.g. upright instructed but held
 *  sideways); a wrong prior can then never stop a session. detected.orientation_match follows the
 *  same rule (gripMatches on the baseline gravity): true / false, or null when undecidable — 'free',
 *  'unknown', or a mismatch that only an unverified sign creates — so the hands-free auto-reject
 *  (orientation_match === false) cannot loop on a wrong sign either. rotation_start / rotation_end
 *  are still the plain readings under the given sign.
 *
 * Details:
 *  - The segmenter is fed the sample exactly as stored (recorder.addMotionRel: relative, rounded),
 *    so re-running online_v1 on the record with GO at timing.go_ms reproduces onset/end indices.
 *  - Every cue played is also logged in streams.cues (ready at t = 0; go/end/error at the time of
 *    the motion sample that triggered them, so the go cue t === timing.go_ms). The trial state is
 *    updated before any caller callback runs (cues.play, notifyCue, onLive), so a callback that
 *    aborts re-entrantly still produces a consistent record.
 *  - Motion samples with a non-finite t, or a t earlier than the previous sample, are dropped (never
 *    stored) and reported once in quality.warnings ('motion_t_invalid', 'motion_t_not_monotonic').
 *  - still_ms / foreperiod_ms describe the hold-still that scheduled the (emitted or pending) GO;
 *    'still_lost' / 'false_start' reset them to null. foreperiod_ms is rounded to 0.01 ms BEFORE it
 *    is used, so GO = first sample with t ≥ still_ms + foreperiod_ms exactly. With a gate that held,
 *    still_ms is the sample at which the gates passed (the segmenter was still since ≥ still_hold_ms
 *    before it, so [still_ms − still_hold_ms, go_ms] is always a still window).
 *  - first_touch_ms = first touchstart whose own time is ≥ go_ms, reported while GO had happened.
 *  - stop_ms = time of the sample that finished the trial (or clock.now() for an abort), raised to
 *    the latest stored time, so still ≤ go ≤ onset ≤ end ≤ end_detected ≤ return_onset ≤
 *    return_end ≤ stop.
 *  - Offline sources: if source.start() returns after delivering samples synchronously but the trial
 *    has not finished (the simulated source ran out of time), the trial finishes as 'interrupted'
 *    with warning 'source_ended' — `result` always resolves.
 *  - source.start() throwing/rejecting before the trial finished → the trial finishes as
 *    'interrupted' (warning 'source_start_failed: …', `result` resolves) AND start() rejects with
 *    that error. start() is idempotent; abort() before start() yields an empty 'aborted' record and
 *    start() then resolves without starting the source.
 *  - Sensor errors (listener.onError) only add 'sensor_error:<code>' warnings; 'no_events' also sets
 *    live.message = 'sensor_stalled'. Warnings about OPTIONAL streams (listener.onWarning, e.g.
 *    'abs_orientation_unavailable' from sensors.ts) are written at most once per session: into this
 *    trial unless an earlier trial of the same session_id that reached GO already carries them
 *    (attempts that never reach GO are discarded by the session layer). Callbacks injected by the
 *    caller (cues, onLive, notifyCue, isHidden, getScreenOrientation) are guarded: a throwing
 *    callback never breaks a recording.
 *  - The quaternion streams' frequency_hz is QUATERNION_NOMINAL_HZ and the magnetometer's
 *    MAGNETOMETER_NOMINAL_HZ (the samples carry no rate: these are what sensors.ts requests).
 *  - live.gyro_dps / acc_ms2 are the smoothed signals, also after the end (the segmenter's are frozen).
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import { ORIENTATIONS, ROUNDING, SCHEMA_VERSION } from './constants';
import { rotationFromGravity, tiltChangeDeg } from './orientationDetect';
import { StreamRecorder, computeQuality, roundValue } from './recorder';
import { MotionSignals, OnlineSegmenter, type RelMotionSample, type SegmenterEvent } from './segmentation';
import type {
  CalibrationResult,
  CueKind,
  MagnetometerSample,
  MotionSample,
  MotionStream,
  OrientationSample,
  PhoneRotation,
  QuaternionSample,
  ScreenSample,
  SensorError,
  SensorListener,
  TouchDetails,
  TrialControllerOptions,
  TrialLiveState,
  TrialMessage,
  TrialPhase,
  TrialRecord,
  TrialStatus,
  Vec3,
} from './types';

export interface TrialController {
  start(): Promise<void>;
  abort(reason?: 'user' | 'interrupted'): void;
  /**
   * `t` in the source clock (performance.now() for real sensors); `details` = radius/force/rotation and
   * the viewport-normalised position of the same touch as x / y (sensors.readTouch builds them).
   */
  addTouch(
    type: 'start' | 'end' | 'cancel',
    n: number,
    x: number | null,
    y: number | null,
    t?: number,
    details?: Partial<TouchDetails> | null,
  ): void;
  readonly result: Promise<TrialRecord>;
  readonly live: TrialLiveState;
  /** time (source clock) of the last motion sample, for UI watchdogs */
  readonly lastSampleAt: number | null;
}

/** detected.rotation_end of a trial without an end uses the mean accg of this window before stop_ms. */
export const END_GRAVITY_WINDOW_MS = 200;
/** QuaternionSample carries no rate: record what the app requests (sensors.ts default, services.ts). */
export const QUATERNION_NOMINAL_HZ = 60;
/** Magnetometer readings carry no rate either (sensors.ts requests 60 Hz). */
export const MAGNETOMETER_NOMINAL_HZ = 60;
/** Return movement (awaitReturn, and the return after the end): smoothed |ω| above this (deg/s) ... */
export const RETURN_GYRO_DPS = 45;
/** ... or smoothed |a_lin| above this (m/s²) ... */
export const RETURN_ACC_MS2 = 1.5;
/** ... continuously for at least this long (ms). */
export const RETURN_SUSTAIN_MS = 100;
/** Grip gate: gravity is averaged over this window before the sample being judged (ms). */
export const GRIP_WINDOW_MS = 300;
/** Grip gate: fewer accg samples than this in the window never block. */
export const GRIP_MIN_SAMPLES = 3;
/** A stillness loss before GO is a COUNTED false start only above one of these (smoothed, deg/s · m/s²). */
export const FALSE_START_GYRO_DPS = 30;
export const FALSE_START_ACC_MS2 = 0.6;
/** ... reached at the loss of stillness or within this long after it (ms). */
export const FALSE_START_WATCH_MS = 300;
/** quality warning: an abort (or the source) ended the recording during the return; the trial is complete. */
export const RETURN_CUT_SHORT = 'return_cut_short';
const MAX_DATE_MS = 8.64e15;

/**
 * The extra listener method sensors.ts calls for problems with OPTIONAL streams (absolute orientation,
 * magnetometer, ...). Not part of SensorListener in types.ts; sources check that it exists.
 */
export interface SensorWarningListener {
  onWarning?(code: string): void;
}

/** session_id → optional-stream warnings already written to a trial of that session that reached GO. */
const sessionWarnings = new Map<string, Set<string>>();

/** Forgets which optional-stream warnings were written (one session, or all). For tests and long-lived pages. */
export function resetSessionWarnings(sessionId?: string): void {
  if (sessionId === undefined) sessionWarnings.clear();
  else sessionWarnings.delete(sessionId);
}

/**
 * True when the arm may still be at the nose after this attempt: a complete trial in which no return
 * movement was seen. The next chained trial should then start with awaitReturn: true.
 */
export function returnMissed(rec: TrialRecord): boolean {
  return rec.outcome.status === 'complete' && (rec.timing.return_onset_ms ?? null) === null;
}

/**
 * True when the calibration's gravity sign is backed by a measured, still upright hold that reads
 * "top edge up" (accg_y·sign > 7 m/s²) under that sign; an engine prior alone is not verified.
 */
export function signVerified(cal: CalibrationResult | null | undefined): boolean {
  const sign = cal?.gravity_sign;
  const up = cal?.upright;
  if ((sign !== 1 && sign !== -1) || !up || up.still !== true) return false;
  const y = up.accg_mean?.[1];
  return typeof y === 'number' && Number.isFinite(y) && y * sign > 7;
}

function errorText(e: unknown): string {
  const s = e instanceof Error ? e.message : String(e);
  return s.length > 200 ? `${s.slice(0, 200)}…` : s;
}

function isoTime(epochMs: number): string {
  const ok = Number.isFinite(epochMs) && Math.abs(epochMs) <= MAX_DATE_MS;
  return new Date(ok ? epochMs : Date.now()).toISOString();
}

/** Mean accg over stored samples with from ≤ t ≤ to (t is non-decreasing); null if none. */
function meanAccg(m: MotionStream, from: number, to: number): Vec3 | null {
  let x = 0;
  let y = 0;
  let z = 0;
  let n = 0;
  for (let i = m.t.length - 1; i >= 0; i--) {
    const t = m.t[i];
    if (t < from) break;
    if (t > to) continue;
    const ax = m.accg_x[i];
    const ay = m.accg_y[i];
    const az = m.accg_z[i];
    if (ax === null || ay === null || az === null) continue;
    x += ax;
    y += ay;
    z += az;
    n++;
  }
  return n > 0 ? [x / n, y / n, z / n] : null;
}

export function createTrialController(opts: TrialControllerOptions): TrialController {
  const { source, cues, params } = opts;
  const clock = source.clock;
  const seg = new OnlineSegmenter(params);
  const signals = new MotionSignals(params);

  const info = ORIENTATIONS[opts.condition.orientation];
  const instructed: PhoneRotation | null = info ? info.rotation : null;
  const screenHint = info ? info.screen : null;
  const detectionSign = opts.calibration?.gravity_sign ?? null;
  const gripGate = opts.gripGate === true && instructed !== null;
  const trustedSign = signVerified(opts.calibration);
  const capRaw = params.return_cap_ms;
  const returnCapMs = typeof capRaw === 'number' && Number.isFinite(capRaw) && capRaw > 0 ? capRaw : 0;

  let recorder: StreamRecorder | null = null;
  let t0 = 0;
  let startPromise: Promise<void> | null = null;
  let sourceStarted = false;
  let inSourceStart = false;
  let syncSamples = 0;
  let finished = false;

  let phase: TrialPhase = 'waiting_still';
  let returnSeen = opts.awaitReturn !== true; // awaitReturn: no GO before a return movement
  let message: TrialMessage | null = returnSeen ? 'hold_still' : 'return_arm';
  let live: TrialLiveState = { phase, t: 0, gyro_dps: 0, acc_ms2: 0, still_progress: 0, message };
  let lastSampleAt: number | null = null;
  let lastMotionT = -Infinity;

  let stillMs: number | null = null;
  let foreperiodMs: number | null = null;
  let goPendingAt: number | null = null;
  let goMs: number | null = null;
  let onsetMs: number | null = null;
  let endMs: number | null = null;
  let endDetectedMs: number | null = null;
  let firstTouchMs: number | null = null;
  let falseStarts = 0;
  let hidden = false;
  const warnings: string[] = [];
  const onceWarnings: string[] = [];
  let screenStart: { type: string | null; angle: number | null } = { type: null, angle: null };

  // gates and the return of the arm
  let gripBlocked = false;
  const gravityWindow: { t: number; g: Vec3 }[] = [];
  let bigRunStart: number | null = null;
  let bigRunCounted = false;
  let returnOnsetMs: number | null = null;
  let returnEndMs: number | null = null;
  let returnStillStart: number | null = null;
  /** false-start policy: a silent loss of stillness (GO was pending) is watched until this rel time */
  let lossWatchUntil: number | null = null;

  let resolveResult!: (r: TrialRecord) => void;
  let rejectResult!: (e: unknown) => void;
  const result = new Promise<TrialRecord>((res, rej) => {
    resolveResult = res;
    rejectResult = rej;
  });

  const warn = (w: string): void => {
    if (!warnings.includes(w)) warnings.push(w);
  };

  /** Runs a caller-supplied callback; a throwing UI/audio callback must not cost the recording. */
  const guarded = (fn: () => void): void => {
    try {
      fn();
    } catch {
      // ignored on purpose (see above)
    }
  };

  const pollHidden = (): void => {
    if (hidden || !opts.isHidden) return;
    guarded(() => {
      if (opts.isHidden?.() === true) hidden = true;
    });
  };

  const readScreen = (): { type: string | null; angle: number | null } => {
    let out: { type: string | null; angle: number | null } = { type: null, angle: null };
    guarded(() => {
      const s = opts.getScreenOrientation?.();
      if (s) {
        out = {
          type: typeof s.type === 'string' ? s.type : null,
          angle: typeof s.angle === 'number' && Number.isFinite(s.angle) ? s.angle : null,
        };
      }
    });
    return out;
  };

  const ensureRecorder = (): StreamRecorder => {
    if (recorder) return recorder;
    const now = clock.now();
    t0 = Number.isFinite(now) ? now : 0;
    recorder = new StreamRecorder(t0);
    screenStart = readScreen();
    return recorder;
  };

  const setLive = (t: number): void => {
    live = {
      phase,
      t,
      gyro_dps: signals.gyro,
      acc_ms2: signals.acc,
      still_progress: seg.live.stillProgress,
      message,
    };
    const onLive = opts.onLive;
    if (onLive) {
      const snapshot = live;
      guarded(() => onLive(snapshot));
    }
  };

  /**
   * Logs a cue in streams.cues at source-clock time tAbs, then plays it. Callers update the trial
   * state BEFORE calling cue()/notify(), so a callback that re-entrantly aborts the trial finds
   * (and records) a consistent state.
   */
  const cue = (kind: CueKind, tAbs: number): void => {
    recorder?.addCue(tAbs, kind);
    guarded(() => cues.play(kind));
  };

  const notify = (kind: CueKind, tAbs: number): void => {
    if (finished) return;
    guarded(() => source.notifyCue?.(kind, tAbs));
  };

  const pickForeperiod = (): number => {
    let r = 0.5;
    guarded(() => {
      const v = opts.rng();
      if (Number.isFinite(v)) r = Math.min(1, Math.max(0, v));
    });
    const fp = params.foreperiod_min_ms + r * (params.foreperiod_max_ms - params.foreperiod_min_ms);
    return roundValue(fp, ROUNDING.time) ?? params.foreperiod_min_ms;
  };

  const cancelGo = (): void => {
    goPendingAt = null;
    stillMs = null;
    foreperiodMs = null;
  };

  /** Message while waiting for stillness (no GO pending). */
  const waitingMessage = (): TrialMessage => (!returnSeen ? 'return_arm' : gripBlocked ? 'wrong_grip' : 'hold_still');

  /** Keeps the accg samples of the last GRIP_WINDOW_MS (only needed by the grip gate). */
  const pushGravity = (rel: RelMotionSample): void => {
    if (rel.accg) gravityWindow.push({ t: rel.t, g: rel.accg });
    while (gravityWindow.length > 0 && gravityWindow[0].t < rel.t - GRIP_WINDOW_MS) gravityWindow.shift();
  };

  /**
   * Does gravity `g` show the instructed grip? true / false, or null when undecidable: no instruction
   * ('free'), an unclear reading, or a mismatch that flipping an UNVERIFIED gravity sign explains.
   */
  const gripMatches = (g: Vec3 | null): boolean | null => {
    if (!g || instructed === null) return null;
    const seen = rotationFromGravity(g, detectionSign, screenHint);
    if (seen === 'unknown') return null;
    if (seen === instructed) return true;
    if (!trustedSign && rotationFromGravity(g, detectionSign === -1 ? 1 : -1, screenHint) === instructed) return null;
    return false;
  };

  /** true when the phone is confidently held in another grip than instructed (the grip gate). */
  const wrongGrip = (): boolean => {
    if (!gripGate || gravityWindow.length < GRIP_MIN_SAMPLES) return false;
    let x = 0;
    let y = 0;
    let z = 0;
    for (const s of gravityWindow) {
      x += s.g[0];
      y += s.g[1];
      z += s.g[2];
    }
    const n = gravityWindow.length;
    return gripMatches([x / n, y / n, z / n]) === false;
  };

  /** The segmenter is 'still': schedule GO if the gates pass (see the module doc). */
  const tryScheduleGo = (relT: number): void => {
    if (!returnSeen) {
      phase = 'waiting_still';
      message = 'return_arm';
      return;
    }
    if (wrongGrip()) {
      gripBlocked = true;
      phase = 'waiting_still';
      message = 'wrong_grip';
      return;
    }
    gripBlocked = false;
    stillMs = relT;
    foreperiodMs = pickForeperiod();
    goPendingAt = relT + foreperiodMs;
    phase = 'foreperiod';
    message = 'hold_still';
  };

  /** Start time of a return-sized movement run, reported once, at the sample that confirms it. */
  const trackBigMotion = (t: number, gyro: number, acc: number): number | null => {
    if (!(gyro > RETURN_GYRO_DPS || acc > RETURN_ACC_MS2)) {
      bigRunStart = null;
      bigRunCounted = false;
      return null;
    }
    if (bigRunStart === null) bigRunStart = t;
    if (!bigRunCounted && t - bigRunStart >= RETURN_SUSTAIN_MS) {
      bigRunCounted = true;
      return bigRunStart;
    }
    return null;
  };

  /** After the return movement: the first sample of stillness held still_hold_ms → return_end_ms. */
  const trackReturnStill = (t: number, gyro: number, acc: number): void => {
    if (!(gyro < params.still_gyro_dps && acc < params.still_acc_ms2)) {
      returnStillStart = null;
      return;
    }
    if (returnStillStart === null) returnStillStart = t;
    if (t - returnStillStart >= params.still_hold_ms) returnEndMs = returnStillStart;
  };

  /** GO (spec: play, notifyCue, addCue, markGo) — state first, callbacks last (see cue()). */
  const emitGo = (relT: number, tAbs: number): void => {
    goPendingAt = null;
    goMs = relT;
    seg.markGo(relT);
    phase = 'armed';
    message = 'go';
    cue('go', tAbs);
    notify('go', tAbs);
  };

  /** Optional-stream warnings: once per session (see the module doc). */
  const applyOnceWarnings = (): void => {
    if (onceWarnings.length === 0) return;
    const sid = opts.ids.session_id;
    const written = sessionWarnings.get(sid);
    for (const w of onceWarnings) if (!written?.has(w)) warn(w);
    if (goMs === null) return; // a discarded attempt does not use up the session's report
    const set = written ?? new Set<string>();
    for (const w of onceWarnings) set.add(w);
    sessionWarnings.set(sid, set);
  };

  const finish = (status: TrialStatus, atRel?: number): void => {
    if (finished) return;
    finished = true;
    if (sourceStarted) guarded(() => source.stop());
    try {
      const rec = ensureRecorder();
      pollHidden();
      applyOnceWarnings();
      let stop = atRel ?? rec.rel(clock.now()) ?? 0;
      const last = rec.lastTime;
      if (last !== null && last > stop) stop = last;
      stop = roundValue(stop, ROUNDING.time) ?? 0;

      const streams = rec.build();
      const quality = computeQuality(streams, hidden, warnings);
      const segmentation = seg.result();
      const gStart = seg.baseline?.gravity ?? null;
      const gEnd =
        endMs !== null && endDetectedMs !== null
          ? meanAccg(streams.motion, endMs, endDetectedMs)
          : meanAccg(streams.motion, stop - END_GRAVITY_WINDOW_MS, stop);
      const rotationStart = gStart ? rotationFromGravity(gStart, detectionSign, screenHint) : 'unknown';
      const rotationEnd = gEnd ? rotationFromGravity(gEnd, detectionSign, screenHint) : 'unknown';
      const screenEnd = readScreen();
      const epoch0 = clock.epochAt(t0);
      const epochStop = clock.epochAt(t0 + stop);
      const ids = opts.ids;

      const record: TrialRecord = {
        record_type: 'trial',
        schema_version: SCHEMA_VERSION,
        app_version: opts.appVersion,
        trial_id: ids.trial_id,
        session_id: ids.session_id,
        participant_id: ids.participant_id,
        study_code: ids.study_code,
        created_at: isoTime(epochStop),
        block_index: ids.block_index,
        trial_index: ids.trial_index,
        trial_in_block: ids.trial_in_block,
        attempt: ids.attempt,
        simulated: opts.simulated,
        condition: { ...opts.condition },
        outcome: { status, accepted: status === 'complete', participant_flag: null, note: null },
        timing: {
          t0_perf_ms: roundValue(t0, ROUNDING.time) ?? 0,
          t0_epoch_ms: roundValue(epoch0, ROUNDING.time) ?? Date.now(),
          still_ms: stillMs,
          go_ms: goMs,
          onset_ms: onsetMs,
          end_ms: endMs,
          end_detected_ms: endDetectedMs,
          stop_ms: stop,
          first_touch_ms: firstTouchMs,
          foreperiod_ms: foreperiodMs,
          false_starts: falseStarts,
          return_onset_ms: returnOnsetMs,
          return_end_ms: returnEndMs,
        },
        segmentation,
        detected: {
          rotation_start: rotationStart,
          rotation_end: rotationEnd,
          tilt_change_deg: roundValue(tiltChangeDeg(gStart, gEnd), ROUNDING.angle),
          orientation_match: gripMatches(gStart),
        },
        quality,
        screen: {
          type_start: screenStart.type,
          angle_start: screenStart.angle,
          type_end: screenEnd.type,
          angle_end: screenEnd.angle,
        },
        streams,
      };
      phase = 'done';
      if (status === 'aborted' || status === 'interrupted') message = null;
      setLive(stop);
      resolveResult(record);
    } catch (e) {
      rejectResult(e);
    }
  };

  /**
   * Ends the trial early (abort, offline source exhausted). During 'returning' the movement and the
   * minimum post-roll are already in the record: the trial is complete, only the return was cut short.
   */
  const finishEarly = (status: 'aborted' | 'interrupted'): void => {
    if (phase === 'returning') {
      warn(RETURN_CUT_SHORT);
      message = 'done';
      finish('complete');
      return;
    }
    finish(status);
  };

  /** An exception inside our own handlers is a bug: keep the data recorded so far, never hang. */
  const fail = (e: unknown): void => {
    warn(`internal_error: ${errorText(e)}`);
    finish('interrupted');
  };

  /** The motion of a stillness loss is big enough for a counted false start. */
  const falseStartMotion = (sig: { gyro: number; acc: number }): boolean =>
    sig.gyro > FALSE_START_GYRO_DPS || sig.acc > FALSE_START_ACC_MS2;

  /** A counted false start: state first, then the error cue (see cue()). */
  const countFalseStart = (tAbs: number): void => {
    lossWatchUntil = null;
    falseStarts++;
    message = 'too_early';
    cue('error', tAbs);
  };

  const handleEvent = (e: SegmenterEvent, tAbs: number, sig: { gyro: number; acc: number }): void => {
    switch (e.type) {
      case 'still':
        lossWatchUntil = null;
        tryScheduleGo(e.t);
        break;
      case 'still_lost':
      case 'false_start': {
        // the false-start policy (module doc): the segmenter's split does not matter here
        const pending = goPendingAt !== null;
        cancelGo();
        phase = 'waiting_still';
        if (pending && falseStartMotion(sig)) {
          countFalseStart(tAbs);
          break;
        }
        message = waitingMessage(); // silent so far
        lossWatchUntil = pending ? e.t + FALSE_START_WATCH_MS : null;
        break;
      }
      case 'onset':
        onsetMs = e.t;
        phase = 'moving';
        message = 'moving';
        break;
      case 'end':
        endMs = e.t;
        endDetectedMs = e.detected_t;
        phase = 'post_roll';
        message = 'return_arm';
        bigRunStart = null;
        bigRunCounted = false;
        cue('end', tAbs);
        notify('end', tAbs);
        break;
      case 'no_movement':
      case 'too_long':
        message = e.type;
        cue('error', tAbs);
        finish(e.type, e.t);
        break;
    }
  };

  const onMotion = (s: MotionSample): void => {
    if (finished || !recorder) return;
    if (inSourceStart) syncSamples++;
    try {
      const tAbs = s.t;
      if (typeof tAbs !== 'number' || !Number.isFinite(tAbs)) {
        warn('motion_t_invalid');
        return;
      }
      if (tAbs < lastMotionT) {
        warn('motion_t_not_monotonic');
        return;
      }
      lastMotionT = tAbs;
      lastSampleAt = tAbs;
      pollHidden();
      const rel = recorder.addMotionRel(s);
      const sig = signals.push(rel);
      if (gripGate) pushGravity(rel);
      const runStart = trackBigMotion(rel.t, sig.gyro, sig.acc);
      if (runStart !== null) {
        if (!returnSeen && goMs === null) returnSeen = true;
        if (endDetectedMs !== null && returnOnsetMs === null && runStart > endDetectedMs) returnOnsetMs = runStart;
      }
      if (returnOnsetMs !== null && returnEndMs === null) trackReturnStill(rel.t, sig.gyro, sig.acc);
      for (const e of seg.push(rel)) {
        handleEvent(e, tAbs, sig);
        if (finished) return;
      }
      if (lossWatchUntil !== null) {
        // a silent loss of stillness turns into a counted false start if the motion grows (module doc)
        if (rel.t > lossWatchUntil) lossWatchUntil = null;
        else if (falseStartMotion(sig)) {
          countFalseStart(tAbs);
          if (finished) return; // the error cue's callback aborted the trial re-entrantly
        }
      }
      if (seg.state === 'still' && goPendingAt === null && goMs === null) tryScheduleGo(rel.t);
      if (goPendingAt !== null && seg.state === 'still' && rel.t >= goPendingAt) {
        emitGo(rel.t, tAbs);
        if (finished) return; // a cue callback aborted the trial re-entrantly
      }
      if ((phase === 'post_roll' || phase === 'returning') && endDetectedMs !== null && rel.t >= endDetectedMs + params.post_roll_ms) {
        if (returnEndMs !== null || rel.t >= endDetectedMs + returnCapMs) {
          message = 'done';
          finish('complete', rel.t);
          return;
        }
        phase = 'returning';
      }
      setLive(rel.t);
    } catch (e) {
      fail(e);
    }
  };

  /** Wraps a listener method: ignored after finishing, internal errors end the trial. */
  const recording = <T>(fn: (rec: StreamRecorder, s: T) => void) =>
    (s: T): void => {
      if (finished || !recorder) return;
      try {
        fn(recorder, s);
      } catch (e) {
        fail(e);
      }
    };

  const listener: SensorListener & SensorWarningListener = {
    onMotion,
    onOrientation: recording<OrientationSample>((rec, s) => {
      pollHidden();
      rec.addOrientation(s);
    }),
    onQuaternion: recording<QuaternionSample>((rec, s) => rec.addQuaternion(s, QUATERNION_NOMINAL_HZ)),
    onOrientationAbs: recording<OrientationSample>((rec, s) => rec.addOrientationAbs(s)),
    onQuaternionAbs: recording<QuaternionSample>((rec, s) => rec.addQuaternionAbs(s, QUATERNION_NOMINAL_HZ)),
    onMagnetometer: recording<MagnetometerSample>((rec, s) => rec.addMagnetometer(s, MAGNETOMETER_NOMINAL_HZ)),
    onScreen: recording<ScreenSample>((rec, s) => rec.addScreen(s)),
    onError(e: SensorError): void {
      if (finished) return;
      warn(`sensor_error:${e?.code ?? 'unknown'}`);
      if (e?.code === 'no_events') {
        message = 'sensor_stalled';
        setLive(live.t);
      }
    },
    onWarning(code: string): void {
      if (finished || typeof code !== 'string' || code === '') return;
      if (!onceWarnings.includes(code)) onceWarnings.push(code);
    },
  };

  /** Source failure before the trial finished: keep a record, then let start() reject. */
  const startFailed = (e: unknown): void => {
    if (finished) return; // a late error after a complete trial does not matter
    warn(`source_start_failed: ${errorText(e)}`);
    finish('interrupted');
    throw e;
  };

  const run = async (): Promise<void> => {
    if (finished) return;
    ensureRecorder();
    pollHidden();
    if (opts.readyCue !== false) cue('ready', t0);
    sourceStarted = true;
    inSourceStart = true;
    let pending: Promise<void>;
    try {
      pending = source.start(listener);
    } catch (e) {
      inSourceStart = false;
      startFailed(e);
      return;
    }
    inSourceStart = false;
    try {
      await pending;
    } catch (e) {
      startFailed(e);
      return;
    }
    if (!finished && syncSamples > 0) {
      warn('source_ended');
      finishEarly('interrupted');
    }
  };

  return {
    start(): Promise<void> {
      if (!startPromise) startPromise = run();
      return startPromise;
    },
    abort(reason: 'user' | 'interrupted' = 'user'): void {
      if (finished) return;
      ensureRecorder();
      finishEarly(reason === 'interrupted' ? 'interrupted' : 'aborted');
    },
    addTouch(
      type: 'start' | 'end' | 'cancel',
      n: number,
      x: number | null,
      y: number | null,
      t?: number,
      details?: Partial<TouchDetails> | null,
    ): void {
      if (finished || !recorder) return;
      try {
        const tAbs = typeof t === 'number' ? t : clock.now();
        const relT = recorder.rel(tAbs);
        if (relT === null) return;
        recorder.addTouch(tAbs, type, n, x, y, details);
        if (type === 'start' && goMs !== null && firstTouchMs === null && relT >= goMs) firstTouchMs = relT;
      } catch (e) {
        fail(e);
      }
    },
    result,
    get live(): TrialLiveState {
      return live;
    },
    get lastSampleAt(): number | null {
      return lastSampleAt;
    },
  };
}
