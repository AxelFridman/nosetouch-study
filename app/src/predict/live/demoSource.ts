/**
 * LiveSource of the desktop demo (?predict=1&sim=1; docs/stage3/APP_SPEC.md section 3). Replays a
 * DemoGenerator's samples in real time through the SAME live path as the real sensors (LiveSample ->
 * engine.push -> UI): a 60 Hz timer emits every sample whose t - t_first is at most the elapsed wall time,
 * with t (and ts) rebased onto performance.now(). No permission prompt, no wake lock, no vibration, no
 * sound. More touches are generated only when the replay runs out (about three minutes at most).
 */
import type { Hand, MotionSample } from '../../core/types';
import { createDemoGenerator, type DemoGenerator } from '../sim/demoScript';
import type { BrowserEngine, LiveSample, LiveSource, SourceFailure } from '../types';

export interface DemoSourceOptions {
  readonly seed: number;
  /** null: the hand is drawn from the seed */
  readonly hand: Hand | null;
  /** wall clock (default performance.now) */
  readonly now?: () => number;
  /** replay period (default 1000/60 ms) */
  readonly periodMs?: number;
}

export interface DemoSource extends LiveSource {
  readonly kind: 'demo';
  readonly hand: Hand;
  readonly seed: number;
  readonly platform: 'ios' | 'android';
}

export function createDemoSource(opts: DemoSourceOptions): DemoSource {
  const gen: DemoGenerator = createDemoGenerator(opts.seed, opts.hand);
  const now = opts.now ?? (() => performance.now());
  const periodMs = opts.periodMs ?? 1000 / 60;
  let timer: ReturnType<typeof setInterval> | null = null;
  let stopped = false;
  let started = false;
  let queue: MotionSample[] = [];
  let qi = 0;
  let tFirst: number | null = null;
  let wallStart = 0;
  let exhausted = false;

  function refill(): boolean {
    if (exhausted) return false;
    const next = gen.nextTouch();
    if (!next) {
      exhausted = true;
      return false;
    }
    queue = queue.slice(qi).concat(next.samples);
    qi = 0;
    return true;
  }

  function pump(onSample: (s: LiveSample) => void): void {
    if (stopped) return;
    const elapsed = now() - wallStart;
    for (;;) {
      if (qi >= queue.length && !refill()) {
        stop(); // every touch replayed: the person stops (the page keeps its last state)
        return;
      }
      const s = queue[qi];
      if (tFirst === null) tFirst = s.t;
      if (s.t - tFirst > elapsed) return;
      qi += 1;
      const shift = wallStart - tFirst;
      onSample({ ...s, t: s.t + shift, ts: s.ts === null ? null : s.ts + shift });
      if (stopped) return;
    }
  }

  function stop(): void {
    stopped = true;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  return {
    kind: 'demo',
    engine: gen.engine as BrowserEngine,
    hand: gen.hand,
    seed: gen.seed,
    platform: gen.platform,
    start(onSample: (s: LiveSample) => void, _onFailure: (reason: SourceFailure) => void): Promise<void> {
      if (started) return Promise.reject(new Error('demo source: start() called twice (sources are single-use)'));
      started = true;
      if (stopped) return Promise.resolve();
      wallStart = now();
      timer = setInterval(() => pump(onSample), periodMs);
      return Promise.resolve();
    },
    stop,
  };
}
