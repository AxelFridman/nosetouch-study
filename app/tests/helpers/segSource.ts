/**
 * Fake SensorSources for the TrialController tests (independent of src/sim).
 *  - 'offline':  delivers ALL samples synchronously inside start() (virtual clock) until stop() is
 *                called from a listener callback or maxDurationMs elapses; calls nothing async.
 *  - 'async':    virtual clock, samples delivered in small batches from setTimeout(0) macrotasks
 *                after start() has resolved (exercises the asynchronous code path quickly).
 *  - 'realtime': setInterval at the sample rate, t = performance.now() (a real-time smoke test).
 * The simulated participant reacts to notifyCue('go') with a minimum-jerk movement after reactionMs,
 * and to notifyCue('end') by bringing the arm back out (the movement reversed, returnMs long,
 * returnAfterEndMs after the cue; null = stays at the nose).
 * Behaviours: normal · false_start (a quick out-and-back jerk during the foreperiod, then normal) ·
 * sway (a slow swayDeg rotation over 500 ms at falseStartAtMs: breaks stillness without being a
 * counted false start, then normal) · no_move (ignores GO) · endless (a rotation that never stops
 * after GO) · never_still (too shaky to ever satisfy the hold-still criterion).
 * preMoves: scripted movements (startMs relative to the first sample), e.g. a return from the nose
 * or a grip correction before GO. extraStreams: also emits the schema 1.2 streams (absolute
 * orientation and quaternion, magnetometer on every tick, one screen orientation change at sample 5).
 * warnings: codes sent through listener.onWarning when the source starts.
 */
import type { CueKind, SensorListener, SensorSource, SourceClock } from '../../src/core/types';
import { SegMotionModel, segRng, type SegModelOptions, type SegMove } from './segSignals';

export type SegBehavior = 'normal' | 'false_start' | 'sway' | 'no_move' | 'endless' | 'never_still';

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
  /** false_start / sway: when it starts, ms after the first sample (inside the foreperiod) */
  falseStartAtMs?: number;
  /** sway: rotation angle (deg) of the slow posture shift (default 7.5 → ≈ 28 deg/s peak) */
  swayDeg?: number;
  /** delay of the return movement after the 'end' cue (ms); null = never returns. Default 350. */
  returnAfterEndMs?: number | null;
  /** duration of the return movement (default 900 ms) */
  returnMs?: number;
  /** scripted movements, startMs relative to the first sample */
  preMoves?: SegMove[];
  /** also emit the schema 1.2 streams (see the module doc) */
  extraStreams?: boolean;
  /** listener.onWarning codes sent at start */
  warnings?: string[];
  /** called after every delivered motion sample (test hooks: aborts, touches, ...) */
  onTick?: (t: number, n: number) => void;
  failStart?: 'reject' | 'throw';
  /** a misbehaving source that keeps delivering after stop() */
  ignoreStop?: boolean;
  /** async mode: samples per macrotask */
  batch?: number;
}

export interface SegFakeSource extends SensorSource {
  readonly truth: { go_t: number | null; onset_t: number | null; end_t: number | null; return_t: number | null };
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
  const returnAfterEnd = o.returnAfterEndMs === undefined ? 350 : o.returnAfterEndMs;
  const returnMs = o.returnMs ?? 900;
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
  const truth: SegFakeSource['truth'] = { go_t: null, onset_t: null, end_t: null, return_t: null };
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
    if (firstT === null) {
      firstT = t;
      for (const m of o.preMoves ?? []) model.addMove({ ...m, startMs: t + m.startMs });
    }
    if (behavior === 'false_start' && !falseStartDone && t >= firstT + (o.falseStartAtMs ?? 650)) {
      falseStartDone = true;
      model.addMove({ startMs: t, durMs: 200, distM: 0.05, angleDeg: 25 });
      model.addMove({ startMs: t + 200, durMs: 200, distM: -0.05, angleDeg: -25 });
    }
    if (behavior === 'sway' && !falseStartDone && t >= firstT + (o.falseStartAtMs ?? 650)) {
      falseStartDone = true;
      model.addMove({ startMs: t, durMs: 500, distM: 0, angleDeg: o.swayDeg ?? 7.5 });
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
    const half = (theta * Math.PI) / 360;
    if (o.quaternion) {
      listener.onQuaternion?.({ t: t + 0.7, q: [Math.sin(half), 0, 0, Math.cos(half)], sensor: 'RelativeOrientationSensor' });
    }
    if (o.extraStreams) {
      listener.onOrientationAbs?.({
        t: t + 0.6,
        ts: t - 0.7,
        alpha: (((theta + 123.4) % 360) + 360) % 360,
        beta: 80 - theta / 4,
        gamma: 2.5,
        absolute: true,
        compass: null,
      });
      listener.onQuaternionAbs?.({ t: t + 0.8, q: [0, 0, Math.sin(half), Math.cos(half)], sensor: 'AbsoluteOrientationSensor' });
      listener.onMagnetometer?.({ t: t + 0.9, x: 20 + theta / 10, y: -30, z: 15.25 });
      if (n === 5) listener.onScreen?.({ t: t + 0.3, type: 'landscape-primary', angle: 90 });
    }
    o.onTick?.(t, n);
  };

  const nextTime = (t0: number, n: number): number =>
    t0 + n * period + (o.jitterMs ? (2 * timing() - 1) * o.jitterMs : 0);

  const sendWarnings = (listener: SensorListener): void => {
    const w = (listener as SensorListener & { onWarning?: (code: string) => void }).onWarning;
    for (const code of o.warnings ?? []) w?.call(listener, code);
  };

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
      sendWarnings(listener);
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
      if (cue === 'end') {
        if (returnAfterEnd === null || truth.end_t === null || truth.return_t !== null) return;
        const start = Math.max(t + returnAfterEnd, truth.end_t);
        truth.return_t = start;
        model.addMove({ startMs: start, durMs: returnMs, distM: -distM, angleDeg: -angleDeg });
        return;
      }
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
