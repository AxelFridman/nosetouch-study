/**
 * Fake SensorSources for the TrialController tests (independent of src/sim).
 *  - 'offline':  delivers ALL samples synchronously inside start() (virtual clock) until stop() is
 *                called from a listener callback or maxDurationMs elapses; calls nothing async.
 *  - 'async':    virtual clock, samples delivered in small batches from setTimeout(0) macrotasks
 *                after start() has resolved (exercises the asynchronous code path quickly).
 *  - 'realtime': setInterval at the sample rate, t = performance.now() (a real-time smoke test).
 * The simulated participant reacts to notifyCue('go') with a minimum-jerk movement after reactionMs.
 * Behaviours: normal · false_start (a quick out-and-back jerk during the foreperiod, then normal) ·
 * no_move (ignores GO) · endless (a rotation that never stops after GO) · never_still (too shaky to
 * ever satisfy the hold-still criterion).
 */
import type { CueKind, SensorListener, SensorSource, SourceClock } from '../../src/core/types';
import { SegMotionModel, segRng, type SegModelOptions } from './segSignals';

export type SegBehavior = 'normal' | 'false_start' | 'no_move' | 'endless' | 'never_still';

export interface SegSourceOptions extends Partial<SegModelOptions> {
  mode: 'offline' | 'async' | 'realtime';
  behavior?: SegBehavior;
  hz?: number;
  /** virtual clock value when start() is called (offline/async) */
  startTime?: number;
  maxDurationMs?: number;
  baseEpochMs?: number;
  reactionMs?: number;
  movementMs?: number;
  distM?: number;
  angleDeg?: number;
  jitterMs?: number;
  orientation?: boolean;
  quaternion?: boolean;
  /** false_start: when the jerk starts, ms after the first sample (inside the foreperiod) */
  falseStartAtMs?: number;
  /** called after every delivered motion sample (test hooks: aborts, touches, ...) */
  onTick?: (t: number, n: number) => void;
  failStart?: 'reject' | 'throw';
  /** a misbehaving source that keeps delivering after stop() */
  ignoreStop?: boolean;
  /** async mode: samples per macrotask */
  batch?: number;
}

export interface SegFakeSource extends SensorSource {
  readonly truth: { go_t: number | null; onset_t: number | null; end_t: number | null };
  readonly cueLog: Array<{ cue: CueKind; t: number }>;
  readonly stopCalls: number;
  readonly delivered: number;
  readonly deliveredAfterStop: number;
  /** resolves when an async/realtime source stops delivering */
  readonly drained: Promise<void>;
}

export const SEG_BASE_EPOCH = 1_790_000_000_000;

export function createSegSource(o: SegSourceOptions): SegFakeSource {
  const behavior = o.behavior ?? 'normal';
  const seed = o.seed ?? 1;
  const hz = o.hz ?? 60;
  const period = 1000 / hz;
  const realtime = o.mode === 'realtime';
  const maxDur = o.maxDurationMs ?? (realtime ? 15000 : 30000);
  const baseEpoch = o.baseEpochMs ?? SEG_BASE_EPOCH;
  const reaction = o.reactionMs ?? 250;
  const movementMs = o.movementMs ?? 1000;
  const distM = o.distM ?? 0.45;
  const angleDeg = o.angleDeg ?? 60;
  const model = new SegMotionModel({
    seed,
    axis: o.axis,
    dir: o.dir,
    up: o.up,
    gravitySign: o.gravitySign,
    gyroNoiseDps: behavior === 'never_still' ? 30 : o.gyroNoiseDps,
    accNoiseMs2: o.accNoiseMs2,
    tremorDps: o.tremorDps,
    acc: o.acc,
    accg: o.accg,
    rr: o.rr,
  });
  const timing = segRng((seed ^ 0x9e3779b9) >>> 0);
  const truth: SegFakeSource['truth'] = { go_t: null, onset_t: null, end_t: null };
  const cueLog: Array<{ cue: CueKind; t: number }> = [];
  let now = realtime ? performance.now() : (o.startTime ?? 1000);
  let firstT: number | null = null;
  let falseStartDone = false;
  let stopped = false;
  let stopCalls = 0;
  let delivered = 0;
  let afterStop = 0;
  let drain!: () => void;
  const drained = new Promise<void>((r) => {
    drain = r;
  });

  const clock: SourceClock = {
    now: () => (realtime ? performance.now() : now),
    epochAt: (t: number) => (realtime ? performance.timeOrigin + t : baseEpoch + t),
  };

  const deliver = (listener: SensorListener, t: number, n: number): void => {
    if (stopped) afterStop++;
    now = t;
    if (firstT === null) firstT = t;
    if (behavior === 'false_start' && !falseStartDone && t >= firstT + (o.falseStartAtMs ?? 650)) {
      falseStartDone = true;
      model.addMove({ startMs: t, durMs: 200, distM: 0.05, angleDeg: 25 });
      model.addMove({ startMs: t + 200, durMs: 200, distM: -0.05, angleDeg: -25 });
    }
    const s = model.sampleAt(t);
    delivered++;
    listener.onMotion({ t, ts: t - 1.234, interval: period, ...s });
    const theta = model.kinematics(t).theta;
    if (o.orientation) {
      listener.onOrientation({
        t: t + 0.5,
        ts: t - 0.75,
        alpha: (((theta % 360) + 360) % 360) as number,
        beta: 80 - theta / 4,
        gamma: 2.5,
        absolute: false,
        compass: null,
      });
    }
    if (o.quaternion) {
      const half = (theta * Math.PI) / 360;
      listener.onQuaternion?.({ t: t + 0.7, q: [Math.sin(half), 0, 0, Math.cos(half)], sensor: 'RelativeOrientationSensor' });
    }
    o.onTick?.(t, n);
  };

  const nextTime = (t0: number, n: number): number =>
    t0 + n * period + (o.jitterMs ? (2 * timing() - 1) * o.jitterMs : 0);

  return {
    kind: 'simulated',
    clock,
    truth,
    cueLog,
    drained,
    get stopCalls() {
      return stopCalls;
    },
    get delivered() {
      return delivered;
    },
    get deliveredAfterStop() {
      return afterStop;
    },
    start(listener: SensorListener): Promise<void> {
      if (o.failStart === 'throw') throw new Error('sensor start failed');
      if (o.failStart === 'reject') return Promise.reject(new Error('permission denied'));
      if (o.mode === 'offline') {
        const t0 = now;
        for (let n = 0; ; n++) {
          if (stopped && !o.ignoreStop) break;
          const t = nextTime(t0, n);
          if (t > t0 + maxDur) break;
          deliver(listener, t, n);
        }
        drain();
        return Promise.resolve();
      }
      if (o.mode === 'async') {
        const t0 = now;
        let n = 0;
        const step = (): void => {
          for (let b = 0; b < (o.batch ?? 5); b++) {
            const t = nextTime(t0, n);
            if ((stopped && !o.ignoreStop) || t > t0 + maxDur) {
              drain();
              return;
            }
            deliver(listener, t, n++);
          }
          setTimeout(step, 0);
        };
        setTimeout(step, 0);
        return Promise.resolve();
      }
      const startedAt = performance.now();
      let n = 0;
      const timer = setInterval(() => {
        const t = performance.now();
        if ((stopped && !o.ignoreStop) || t > startedAt + maxDur) {
          clearInterval(timer);
          drain();
          return;
        }
        deliver(listener, t, n++);
      }, period);
      return Promise.resolve();
    },
    stop(): void {
      stopCalls++;
      stopped = true;
    },
    notifyCue(cue: CueKind, t: number): void {
      cueLog.push({ cue, t });
      if (cue !== 'go' || truth.go_t !== null) return;
      truth.go_t = t;
      if (behavior === 'no_move' || behavior === 'never_still') return;
      const onset = t + reaction;
      truth.onset_t = onset;
      if (behavior === 'endless') {
        model.spin = { fromMs: onset, dps: 70 };
        return;
      }
      model.addMove({ startMs: onset, durMs: movementMs, distM, angleDeg });
      truth.end_t = onset + movementMs;
    },
  };
}
