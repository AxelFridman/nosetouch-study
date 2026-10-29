/**
 * One sensing session of the page: a LiveSource feeding a SessionEngine (docs/stage3/APP_SPEC.md 4.2).
 *  - every LiveSample goes to engine.push as it arrives (the engine is driven by sample times only);
 *  - a gap in the data (the page hidden, or more than GAP_MS between two samples) calls notifyGap():
 *    while watching the engine starts a new stream (onRestarted), a final answer stays;
 *  - snapshots reach the UI at most once per animation frame (the engine may emit 60 times a second);
 *  - reset() (Again / Start over) starts a new stream with the same sensors, so no new permission is
 *    needed. The demo instead replays a new script, with a new engine because its platform (and so the
 *    gravity sign of the normalisation) may change between scripts.
 * No storage and no network: nothing in here leaves the page.
 */
import type { EngineSnapshot, LiveSample, LiveSource, ModelId, SessionEngine, SourceFailure } from '../types';
import type { EngineBinding } from './engine';

/** a pause between two samples longer than this restarts a stream that is still watching (APP_SPEC 4.2 step 7) */
export const GAP_MS = 1000;

export interface LiveSessionOptions {
  readonly binding: EngineBinding;
  readonly driver: ModelId;
  /** the source of run 0, 1, 2... (a new one per reset when restartSourceOnReset) */
  readonly makeSource: (run: number) => LiveSource;
  readonly restartSourceOnReset: boolean;
  /** frame-throttled snapshots */
  readonly onSnapshot: (s: EngineSnapshot) => void;
  readonly onFailure: (reason: SourceFailure) => void;
  /** a new attempt was frozen (k = 1, 2...): the page vibrates once on Android */
  readonly onAttempt?: (k: number) => void;
  /** a watching stream was restarted by a gap */
  readonly onRestarted?: () => void;
  /** default: requestAnimationFrame (setTimeout 16 ms without it) */
  readonly schedule?: (fn: () => void) => void;
  /** listen to document visibility (default true when a document exists) */
  readonly watchVisibility?: boolean;
}

/** wall-clock measures of the current run (the &debug=1 panel): they show what the phone's speed does to sampling */
export interface SessionTiming {
  /** the longest engine.push() (ms): one devicemotion event, including a tick that froze an attempt */
  readonly maxPushMs: number;
  /** the largest pause between two consecutive samples (ms); above 100 ms the stream has a gap */
  readonly maxGapMs: number;
}

export interface LiveSession {
  readonly engine: SessionEngine;
  readonly source: LiveSource;
  start(): Promise<void>;
  reset(): void;
  setDriver(id: ModelId): void;
  /** stops the sensors (an answer is on screen: nothing more is read); reset() starts them again */
  stopSource(): void;
  timing(): SessionTiming;
  dispose(): void;
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function defaultSchedule(fn: () => void): void {
  const raf = (globalThis as { requestAnimationFrame?: (cb: () => void) => number }).requestAnimationFrame;
  if (typeof raf === 'function') raf(() => fn());
  else setTimeout(fn, 16);
}

export function createLiveSession(o: LiveSessionOptions): LiveSession {
  const schedule = o.schedule ?? defaultSchedule;
  let run = 0;
  let driver = o.driver;
  let disposed = false;
  let lastT: number | null = null;
  let pending: EngineSnapshot | null = null;
  let scheduled = false;
  let prevAttempts = 0;
  let source: LiveSource;
  let engine: SessionEngine;
  let unsubscribe: () => void = () => undefined;
  let maxPushMs = 0;
  let maxGapMs = 0;
  let broken = false;

  const flush = (): void => {
    scheduled = false;
    if (disposed || !pending) return;
    const s = pending;
    pending = null;
    o.onSnapshot(s);
  };

  const onEngineSnapshot = (s: EngineSnapshot): void => {
    if (s.attempts.length > prevAttempts) o.onAttempt?.(s.attempts.length);
    prevAttempts = s.attempts.length;
    pending = s;
    if (!scheduled) {
      scheduled = true;
      schedule(flush);
    }
  };

  const gap = (): void => {
    const snap = engine.snapshot();
    const restarting = snap.phase === 'watching' && snap.nSamples > 0;
    engine.notifyGap();
    if (restarting) o.onRestarted?.();
  };

  const onSample = (s: LiveSample): void => {
    if (disposed || broken) return;
    if (lastT !== null) {
      const dt = s.t - lastT;
      if (dt > maxGapMs && dt <= GAP_MS) maxGapMs = dt;
      if (dt > GAP_MS) gap();
    }
    lastT = s.t;
    const t0 = now();
    try {
      engine.push(s);
    } catch {
      // never expected (the parity tests replay every recording): a visible message instead of a page that
      // silently stays on 'watching'
      broken = true;
      source.stop();
      o.onFailure('engine_error');
      return;
    }
    const ms = now() - t0;
    if (ms > maxPushMs) maxPushMs = ms;
  };

  const onFailure = (reason: SourceFailure): void => {
    if (!disposed) o.onFailure(reason);
  };

  function attach(): void {
    source = o.makeSource(run);
    engine = o.binding.createSessionEngine({
      normalization: o.binding.normalization(source.engine),
      models: o.binding.models(),
      driver,
    });
    prevAttempts = 0;
    lastT = null;
    maxPushMs = 0;
    maxGapMs = 0;
    unsubscribe = engine.subscribe(onEngineSnapshot);
  }

  const onVisibility = (): void => {
    if (disposed) return;
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      gap();
      lastT = null;
    }
  };
  const watchVisibility = (o.watchVisibility ?? true) && typeof document !== 'undefined';
  if (watchVisibility) document.addEventListener('visibilitychange', onVisibility);

  attach();

  const session: LiveSession = {
    get engine() {
      return engine;
    },
    get source() {
      return source;
    },
    start(): Promise<void> {
      return source.start(onSample, onFailure);
    },
    reset(): void {
      if (disposed) return;
      if (o.restartSourceOnReset) {
        source.stop();
        unsubscribe();
        engine.dispose();
        run += 1;
        attach();
        onEngineSnapshot(engine.snapshot());
        void source.start(onSample, onFailure);
        return;
      }
      lastT = null;
      prevAttempts = 0;
      maxPushMs = 0;
      maxGapMs = 0;
      engine.reset();
    },
    setDriver(id: ModelId): void {
      driver = id;
      if (!disposed) engine.setDriver(id);
    },
    stopSource(): void {
      source.stop();
    },
    timing(): SessionTiming {
      return { maxPushMs, maxGapMs };
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (watchVisibility) document.removeEventListener('visibilitychange', onVisibility);
      source.stop();
      unsubscribe();
      engine.dispose();
    },
  };
  return session;
}
