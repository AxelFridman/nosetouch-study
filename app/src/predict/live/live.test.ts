/**
 * The live adapter: the first-second sensor checks, the preflight, the session controller (gaps, resets,
 * frame throttling) and the wired engine's SessionEngine contract (the real engine: detect/ + models/ +
 * pipeline/). The engine's parity tests live with detect/ and in predict/parity.test.ts.
 */
import { describe, expect, it, vi } from 'vitest';
import type { MotionSample, SensorCapabilities } from '../../core/types';
import { makeDemoScript, tiltSample } from '../sim/demoScript';
import type { EngineSnapshot, LiveSample, LiveSource, SourceFailure } from '../types';
import { createSensorCheck, failureOfSensorError, SENSOR_CHECK_MS } from './browserSource';
import { binding } from './engine';

/**
 * The session tests need a known answer on simulated touches, so they drive with lane m4's classifier: the
 * final CNN reads the simulator's touches with the wrong sign (see e2e/predict.spec.ts and docs/stage3/README.md),
 * m4, PHYS6 and the yaw reference do not.
 */
const wired = binding;
import { demoPreflight, preflightFrom } from './preflight';
import { createLiveSession, GAP_MS } from './session';

const sample = (t: number, over: Partial<MotionSample> = {}): MotionSample => ({
  t,
  ts: t,
  acc: [0, 0, 0],
  accg: [0, 9.2, 3.3],
  rr: { alpha: 0.3, beta: -0.2, gamma: 0.1 },
  interval: 16,
  ...over,
});

describe('first-second sensor checks', () => {
  it('passes a phone with both sensors', () => {
    const check = createSensorCheck();
    expect(check(sample(0))).toBeNull();
    expect(check(sample(SENSOR_CHECK_MS))).toBe('ok');
    expect(check(sample(SENSOR_CHECK_MS + 16, { rr: null }))).toBe('ok');
  });

  it('stops without a gyroscope or an accelerometer in the whole first second', () => {
    const gyro = createSensorCheck();
    for (let t = 0; t < SENSOR_CHECK_MS; t += 16) expect(gyro(sample(t, { rr: null }))).toBeNull();
    expect(gyro(sample(SENSOR_CHECK_MS, { rr: null }))).toBe('no_gyroscope');
    const acc = createSensorCheck();
    acc(sample(0, { accg: null }));
    expect(acc(sample(SENSOR_CHECK_MS, { accg: null }))).toBe('no_accelerometer');
  });

  it('maps the collector source errors to SourceFailure', () => {
    expect(failureOfSensorError('insecure_context')).toBe('insecure_context');
    expect(failureOfSensorError('not_supported')).toBe('not_supported');
    expect(failureOfSensorError('permission_denied')).toBe('permission_denied');
    expect(failureOfSensorError('no_events')).toBe('no_events');
    expect(failureOfSensorError('sensor_error')).toBe('no_events');
  });
});

describe('preflight', () => {
  const caps = (over: Partial<SensorCapabilities> = {}): SensorCapabilities => ({
    secure_context: true,
    device_motion: true,
    device_orientation: true,
    permission_api: true,
    permission_state: 'unknown',
    generic_relative_orientation: false,
    wake_lock: true,
    vibrate: false,
    share_files: false,
    measured: null,
    ...over,
  });

  it('an iPhone: ok, needs the permission tap, webkit sign', () => {
    expect(preflightFrom(caps(), { platform: 'ios', engine: 'webkit', in_app_browser: null })).toEqual({
      ok: true,
      failure: null,
      engine: 'webkit',
      needsPermission: true,
      permission: 'unknown',
      wakeLock: true,
      vibrate: false,
      desktop: false,
    });
  });

  it('blocks an insecure page, an in-app browser and a browser without DeviceMotionEvent', () => {
    expect(preflightFrom(caps({ secure_context: false }), { platform: 'android', engine: 'blink', in_app_browser: null }).failure).toBe('insecure_context');
    expect(preflightFrom(caps(), { platform: 'android', engine: 'blink', in_app_browser: 'instagram' }).failure).toBe('in_app_browser');
    const desk = preflightFrom(caps({ device_motion: false, permission_api: false }), { platform: 'other', engine: 'gecko', in_app_browser: null });
    expect(desk.failure).toBe('not_supported');
    expect(desk.desktop).toBe(true);
    expect(preflightFrom(caps(), { platform: 'android', engine: undefined, in_app_browser: null }).engine).toBe('unknown');
  });

  it('the demo never prompts, vibrates or holds a wake lock', () => {
    expect(demoPreflight('webkit')).toMatchObject({ ok: true, needsPermission: false, wakeLock: false, vibrate: false });
  });
});

/** A LiveSource whose samples the test pushes by hand. */
function manualSource(engine: LiveSource['engine'] = 'blink') {
  let push: ((s: LiveSample) => void) | null = null;
  let failure: ((r: SourceFailure) => void) | null = null;
  const src: LiveSource & { emit(s: LiveSample): void; fail(r: SourceFailure): void; stopped: boolean } = {
    kind: 'browser',
    engine,
    stopped: false,
    start(onSample, onFailure) {
      push = onSample;
      failure = onFailure;
      return Promise.resolve();
    },
    stop() {
      src.stopped = true;
    },
    emit(s) {
      push?.(s);
    },
    fail(r) {
      failure?.(r);
    },
  };
  return src;
}

describe('live session', () => {
  it('feeds the engine, throttles snapshots to frames, reports attempts and the final answer', async () => {
    const script = makeDemoScript(7, 'right');
    const frames: (() => void)[] = [];
    const snaps: EngineSnapshot[] = [];
    const attempts: number[] = [];
    const src = manualSource(script.engine);
    const live = createLiveSession({
      binding,
      driver: 'm4',
      makeSource: () => src,
      restartSourceOnReset: false,
      onSnapshot: (s) => snaps.push(s),
      onFailure: () => undefined,
      onAttempt: (k) => attempts.push(k),
      schedule: (fn) => frames.push(fn),
      watchVisibility: false,
    });
    await live.start();
    for (const s of script.samples.slice(0, 120)) src.emit(s);
    expect(frames).toHaveLength(1); // 120 samples, one frame scheduled
    frames.shift()!();
    expect(snaps).toHaveLength(1);
    expect(snaps[0].nSamples).toBe(120);
    for (const s of script.samples.slice(120)) src.emit(s);
    expect(attempts[0]).toBe(1);
    expect(live.engine.snapshot().phase).toBe('final');
    expect(live.engine.snapshot().decision.hand).toBe('right');
    live.dispose();
    expect(src.stopped).toBe(true);
  });

  it('a pause of more than a second restarts a stream that is still watching', async () => {
    const src = manualSource();
    const onRestarted = vi.fn();
    const live = createLiveSession({
      binding,
      driver: 'cnn',
      makeSource: () => src,
      restartSourceOnReset: false,
      onSnapshot: () => undefined,
      onFailure: () => undefined,
      onRestarted,
      schedule: () => undefined,
      watchVisibility: false,
    });
    await live.start();
    for (let t = 0; t <= 2000; t += 16) src.emit(sample(1000 + t));
    expect(live.engine.snapshot().nSamples).toBeGreaterThan(100);
    src.emit(sample(3000 + 16 + GAP_MS + 1));
    expect(onRestarted).toHaveBeenCalledOnce();
    expect(live.engine.snapshot().nSamples).toBe(1); // the new stream starts with this sample
    expect(live.engine.snapshot().lastEvent?.type).toBe('gap');
    live.dispose();
  });

  it('the demo replays a new script on reset (a new source and a new engine); real sensors keep theirs', async () => {
    const made: number[] = [];
    const sources: ReturnType<typeof manualSource>[] = [];
    const live = createLiveSession({
      binding,
      driver: 'm4',
      makeSource: (run) => {
        made.push(run);
        const s = manualSource(run === 0 ? 'webkit' : 'blink');
        sources.push(s);
        return s;
      },
      restartSourceOnReset: true,
      onSnapshot: () => undefined,
      onFailure: () => undefined,
      schedule: () => undefined,
      watchVisibility: false,
    });
    await live.start();
    sources[0].emit(sample(10));
    const first = live.engine;
    live.reset();
    expect(made).toEqual([0, 1]);
    expect(sources[0].stopped).toBe(true);
    expect(live.engine).not.toBe(first);
    expect(live.engine.snapshot().driver).toBe('m4');
    live.setDriver('yaw');
    expect(live.engine.snapshot().driver).toBe('yaw');
    live.dispose();
  });

  it('forwards a source failure once, and nothing after dispose', async () => {
    const src = manualSource();
    const onFailure = vi.fn();
    const live = createLiveSession({
      binding,
      driver: 'cnn',
      makeSource: () => src,
      restartSourceOnReset: false,
      onSnapshot: () => undefined,
      onFailure,
      schedule: () => undefined,
      watchVisibility: false,
    });
    await live.start();
    src.fail('no_gyroscope');
    expect(onFailure).toHaveBeenCalledWith('no_gyroscope');
    live.dispose();
    src.fail('no_events');
    expect(onFailure).toHaveBeenCalledTimes(1);
  });
});

describe('the wired engine (SessionEngine contract)', () => {
  const engineFor = (engine: LiveSource['engine']) =>
    binding.createSessionEngine({ normalization: binding.normalization(engine), models: binding.models(), driver: 'm4' });

  it('the wired binding offers four models in registry order, the CNN by default, and the engine prior sign', () => {
    expect(wired.specs.map((m) => m.id)).toEqual(['cnn', 'm4', 'phys6', 'yaw']);
    expect(wired.specs.filter((m) => m.isDefault).map((m) => m.id)).toEqual(['cnn']);
    expect(wired.normalization('webkit').gravitySign).toBe(-1);
    expect(wired.normalization('blink').gravitySign).toBe(1);
    expect(wired.normalization('unknown').gravitySign).toBe(1);
    expect(wired.kind).toBe('real');
    expect(wired.specs.every((m) => /^[0-9a-f]{64}$/.test(m.sha256))).toBe(true);
  });

  it('freezes attempts, compares every model after the same attempts, stops by the driver, ignores samples once final', () => {
    for (const hand of ['left', 'right'] as const) {
      const script = makeDemoScript(7, hand);
      const eng = engineFor(script.engine);
      const events: string[] = [];
      eng.subscribe((s) => {
        if (s.lastEvent && events[events.length - 1] !== s.lastEvent.type) events.push(s.lastEvent.type);
      });
      for (const x of script.samples) eng.push(x); // one sample at a time, like the sensors
      const s = eng.snapshot();
      expect(s.phase).toBe('final');
      expect(s.decision.status).toBe('confident');
      expect(s.decision.hand).toBe(hand);
      expect(s.decision.answered).toBe(true);
      expect(s.compare.map((c) => c.modelId)).toEqual(['cnn', 'm4', 'phys6', 'yaw']);
      expect(s.compare.every((c) => c.k === s.decision.nUsed)).toBe(true);
      expect(s.attempts.every((a) => a.frozenAtMs >= a.bout.decidedMs + 1000)).toBe(true);
      expect(events).toContain('attempt');
      const n = s.nSamples;
      eng.push({ ...script.samples[script.samples.length - 1], t: script.samples[script.samples.length - 1].t + 20 });
      expect(eng.snapshot().nSamples).toBe(n);
      // a driver switch recomputes from the frozen attempts at once
      eng.setDriver('yaw');
      expect(eng.snapshot().decision.modelId).toBe('yaw');
      expect(eng.snapshot().driver).toBe('yaw');
      eng.reset();
      expect(eng.snapshot().phase).toBe('watching');
      expect(eng.snapshot().attempts).toHaveLength(0);
      expect(eng.snapshot().driver).toBe('yaw');
      eng.dispose();
      expect(eng.snapshot().phase).toBe('stopped');
    }
  });

  it('a screen-away rest pose ends in the grip phase, with no answer', () => {
    const script = makeDemoScript(7, 'right', 2);
    const eng = engineFor(script.engine);
    // turn the whole recording by 180 deg about the device x axis: upside down, the screen facing away
    eng.push(script.samples.map((s) => tiltSample(tiltSample(s, 90, false), 90, false)));
    const s = eng.snapshot();
    expect(s.phase).toBe('grip');
    expect(s.decision.status).toBe('grip');
    expect(s.decision.answered).toBe(false);
    expect(s.restUpZ).toBeLessThan(0);
  });

  it('without a touch the stream ends at maxStreamMs: no_attempt, an abstention', () => {
    const eng = binding.createSessionEngine({ normalization: binding.normalization('blink'), models: binding.models(), driver: 'cnn', maxStreamMs: 3000 });
    for (let t = 0; t <= 3200; t += 16) eng.push(sample(t));
    const s = eng.snapshot();
    expect(s.phase).toBe('final');
    expect(s.decision.status).toBe('no_attempt');
    expect(s.decision.answered).toBe(false);
  });

  it('notifyGap resets only while watching', () => {
    const eng = engineFor('blink');
    for (let t = 0; t <= 500; t += 16) eng.push(sample(t));
    eng.notifyGap();
    expect(eng.snapshot().nSamples).toBe(0);
    expect(eng.snapshot().lastEvent?.type).toBe('gap');
  });
});
