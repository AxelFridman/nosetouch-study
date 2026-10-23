/**
 * Runs ONE trial: records all sensor streams, drives the OnlineSegmenter, emits cues and builds
 * the TrialRecord. Entirely driven by incoming motion samples (no setTimeout for trial logic), so it
 * is deterministic with a simulated source and runs in Node.
 *
 * Flow:
 *  start(): t0 = source.clock.now(); recorder = new StreamRecorder(t0); cue 'ready';
 *           source.start(listener)  (offline simulated sources run the whole trial inside this call).
 *  each motion sample → recorder.addMotion → segmenter.push(rel sample) → handle events:
 *    'still'        → pick foreperiod = foreperiod_min + rng()·(max−min); phase 'foreperiod'
 *    sample t ≥ still_t + foreperiod while segmenter.state==='still' → GO: cues.play('go'),
 *                     source.notifyCue?.('go', abs t), recorder.addCue, segmenter.markGo(rel t); phase 'armed'
 *    'still_lost'   → cancel pending GO; phase 'waiting_still' (message 'hold_still')
 *    'false_start'  → cancel pending GO; cues.play('error'); falseStarts++; message 'too_early'
 *    'onset'        → phase 'moving' (message 'moving')
 *    'end'          → cues.play('end'); notifyCue('end'); phase 'post_roll' (message 'hold_at_nose' → 'done')
 *    'no_movement' / 'too_long' → cues.play('error'); finish with that status
 *    post_roll: when rel t ≥ end.detected_t + post_roll_ms → finish('complete')
 *  Touch events are recorded via addTouch(); first touchstart after GO → timing.first_touch_ms.
 *  abort('user') → finish('aborted'); abort('interrupted') (page hidden etc.) → finish('interrupted').
 *  finish(): source.stop(); build streams; quality via computeQuality (hidden_during from isHidden
 *    polled on every sample); detected.* via rotationFromGravity on baseline gravity (start) and the
 *    mean accg of the last 200 ms before stop (end), compared with ORIENTATIONS[cond].rotation;
 *    outcome.accepted = (status === 'complete'); resolve `result`.
 *  The trial_id etc. come from opts.ids. created_at = ISO time of finish (source.clock.epochAt).
 *  A watchdog is NOT part of this class: the UI calls abort('interrupted') if no samples arrive.
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
 *    is used, so GO = first sample with t ≥ still_ms + foreperiod_ms exactly.
 *  - first_touch_ms = first touchstart whose own time is ≥ go_ms, reported while GO had happened.
 *  - stop_ms = time of the sample that finished the trial (or clock.now() for an abort), raised to
 *    the latest stored time, so still ≤ go ≤ onset ≤ end ≤ end_detected ≤ stop.
 *  - Offline sources: if source.start() returns after delivering samples synchronously but the trial
 *    has not finished (the simulated source ran out of time), the trial finishes as 'interrupted'
 *    with warning 'source_ended' — `result` always resolves.
 *  - source.start() throwing/rejecting before the trial finished → the trial finishes as
 *    'interrupted' (warning 'source_start_failed: …', `result` resolves) AND start() rejects with
 *    that error. start() is idempotent; abort() before start() yields an empty 'aborted' record and
 *    start() then resolves without starting the source.
 *  - Sensor errors (listener.onError) only add 'sensor_error:<code>' warnings; 'no_events' also sets
 *    live.message = 'sensor_stalled'. Callbacks injected by the caller (cues, onLive, notifyCue,
 *    isHidden, getScreenOrientation) are guarded: a throwing callback never breaks a recording.
 *  - The quaternion stream's frequency_hz is QUATERNION_NOMINAL_HZ (QuaternionSample carries no rate).
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import { ORIENTATIONS, ROUNDING, SCHEMA_VERSION } from './constants';
import { rotationFromGravity, tiltChangeDeg } from './orientationDetect';
import { StreamRecorder, computeQuality, roundValue } from './recorder';
import { OnlineSegmenter, type SegmenterEvent } from './segmentation';
import type {
  CueKind,
  MotionSample,
  MotionStream,
  OrientationSample,
  QuaternionSample,
  SensorError,
  SensorListener,
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
  /** `t` in the source clock (performance.now() for real sensors) */
  addTouch(type: 'start' | 'end' | 'cancel', n: number, x: number | null, y: number | null, t?: number): void;
  readonly result: Promise<TrialRecord>;
  readonly live: TrialLiveState;
  /** time (source clock) of the last motion sample, for UI watchdogs */
  readonly lastSampleAt: number | null;
}

/** detected.rotation_end uses the mean accg of this window before stop_ms. */
export const END_GRAVITY_WINDOW_MS = 200;
/** QuaternionSample carries no rate: record what the app requests (sensors.ts default, services.ts). */
export const QUATERNION_NOMINAL_HZ = 60;
const MAX_DATE_MS = 8.64e15;

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

  let recorder: StreamRecorder | null = null;
  let t0 = 0;
  let startPromise: Promise<void> | null = null;
  let sourceStarted = false;
  let inSourceStart = false;
  let syncSamples = 0;
  let finished = false;

  let phase: TrialPhase = 'waiting_still';
  let message: TrialMessage | null = 'hold_still';
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
  let screenStart: { type: string | null; angle: number | null } = { type: null, angle: null };

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
    const l = seg.live;
    live = { phase, t, gyro_dps: l.gyro, acc_ms2: l.acc, still_progress: l.stillProgress, message };
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

  const finish = (status: TrialStatus, atRel?: number): void => {
    if (finished) return;
    finished = true;
    if (sourceStarted) guarded(() => source.stop());
    try {
      const rec = ensureRecorder();
      pollHidden();
      let stop = atRel ?? rec.rel(clock.now()) ?? 0;
      const last = rec.lastTime;
      if (last !== null && last > stop) stop = last;
      stop = roundValue(stop, ROUNDING.time) ?? 0;

      const streams = rec.build();
      const quality = computeQuality(streams, hidden, warnings);
      const segmentation = seg.result();
      const info = ORIENTATIONS[opts.condition.orientation];
      const instructed = info ? info.rotation : null;
      const hint = info ? info.screen : null;
      const sign = opts.calibration?.gravity_sign ?? null;
      const gStart = seg.baseline?.gravity ?? null;
      const gEnd = meanAccg(streams.motion, stop - END_GRAVITY_WINDOW_MS, stop);
      const rotationStart = gStart ? rotationFromGravity(gStart, sign, hint) : 'unknown';
      const rotationEnd = gEnd ? rotationFromGravity(gEnd, sign, hint) : 'unknown';
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
        },
        segmentation,
        detected: {
          rotation_start: rotationStart,
          rotation_end: rotationEnd,
          tilt_change_deg: roundValue(tiltChangeDeg(gStart, gEnd), ROUNDING.angle),
          orientation_match: rotationStart !== 'unknown' && instructed !== null ? rotationStart === instructed : null,
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

  /** An exception inside our own handlers is a bug: keep the data recorded so far, never hang. */
  const fail = (e: unknown): void => {
    warn(`internal_error: ${errorText(e)}`);
    finish('interrupted');
  };

  const handleEvent = (e: SegmenterEvent, tAbs: number): void => {
    switch (e.type) {
      case 'still':
        stillMs = e.t;
        foreperiodMs = pickForeperiod();
        goPendingAt = e.t + foreperiodMs;
        phase = 'foreperiod';
        message = 'hold_still';
        break;
      case 'still_lost':
        cancelGo();
        phase = 'waiting_still';
        message = 'hold_still';
        break;
      case 'false_start':
        cancelGo();
        falseStarts++;
        phase = 'waiting_still';
        message = 'too_early';
        cue('error', tAbs);
        break;
      case 'onset':
        onsetMs = e.t;
        phase = 'moving';
        message = 'moving';
        break;
      case 'end':
        endMs = e.t;
        endDetectedMs = e.detected_t;
        phase = 'post_roll';
        message = 'hold_at_nose';
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
      for (const e of seg.push(rel)) {
        handleEvent(e, tAbs);
        if (finished) return;
      }
      if (goPendingAt !== null && seg.state === 'still' && rel.t >= goPendingAt) {
        emitGo(rel.t, tAbs);
        if (finished) return; // a cue callback aborted the trial re-entrantly
      }
      if (phase === 'post_roll' && endDetectedMs !== null && rel.t >= endDetectedMs + params.post_roll_ms) {
        message = 'done';
        finish('complete', rel.t);
        return;
      }
      setLive(rel.t);
    } catch (e) {
      fail(e);
    }
  };

  const listener: SensorListener = {
    onMotion,
    onOrientation(s: OrientationSample): void {
      if (finished || !recorder) return;
      try {
        pollHidden();
        recorder.addOrientation(s);
      } catch (e) {
        fail(e);
      }
    },
    onQuaternion(s: QuaternionSample): void {
      if (finished || !recorder) return;
      try {
        recorder.addQuaternion(s, QUATERNION_NOMINAL_HZ);
      } catch (e) {
        fail(e);
      }
    },
    onError(e: SensorError): void {
      if (finished) return;
      warn(`sensor_error:${e?.code ?? 'unknown'}`);
      if (e?.code === 'no_events') {
        message = 'sensor_stalled';
        setLive(live.t);
      }
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
    cue('ready', t0);
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
      finish('interrupted');
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
      finish(reason === 'interrupted' ? 'interrupted' : 'aborted');
    },
    addTouch(type: 'start' | 'end' | 'cancel', n: number, x: number | null, y: number | null, t?: number): void {
      if (finished || !recorder) return;
      try {
        const tAbs = typeof t === 'number' ? t : clock.now();
        const relT = recorder.rel(tAbs);
        if (relT === null) return;
        recorder.addTouch(tAbs, type, n, x, y);
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
