import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createBrowserSensorSource,
  detectCapabilities,
  getDeviceInfo,
  getScreenOrientation,
  measureRates,
  NO_EVENTS_TIMEOUT_MS,
  parseUserAgent,
  rateSummary,
  requestMotionPermission,
} from '../src/core/sensors';
import type {
  MotionSample,
  OrientationSample,
  QuaternionSample,
  SensorError,
  SensorListener,
  SensorSource,
} from '../src/core/types';
import {
  FakeDocument,
  fakeMotionInterfaces,
  fakeOrientationSensorClass,
  fakePermissionsApi,
  motionEvent,
  orientationEvent,
  TrackingTarget,
  type PermissionBehavior,
} from './helpers/sensorsFakes';
import { expectValidDef } from './helpers/sensorsSchema';

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const FAKE_TIMERS: NonNullable<Parameters<typeof vi.useFakeTimers>[0]> = {
  toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date', 'performance'],
};

function collector() {
  const motion: MotionSample[] = [];
  const orientation: OrientationSample[] = [];
  const quaternion: QuaternionSample[] = [];
  const errors: SensorError[] = [];
  const listener: SensorListener = {
    onMotion: (s) => motion.push(s),
    onOrientation: (s) => orientation.push(s),
    onQuaternion: (s) => quaternion.push(s),
    onError: (e) => errors.push(e),
  };
  return { motion, orientation, quaternion, errors, listener };
}

interface BrowserSetup {
  secure?: boolean;
  webkit?: boolean;
  motionPermission?: PermissionBehavior | null;
  orientationPermission?: PermissionBehavior | null;
  navigator?: Record<string, unknown>;
  sensorClass?: unknown;
  noMotionInterface?: boolean;
}

function installBrowser(o: BrowserSetup = {}) {
  const win = new TrackingTarget();
  const ifaces = fakeMotionInterfaces({
    motionPermission: o.motionPermission ?? null,
    orientationPermission: o.orientationPermission ?? null,
    webkit: o.webkit,
  });
  const doc = new FakeDocument();
  vi.stubGlobal('window', win);
  vi.stubGlobal('isSecureContext', o.secure ?? true);
  if (!o.noMotionInterface) vi.stubGlobal('DeviceMotionEvent', ifaces.DeviceMotionEvent);
  vi.stubGlobal('DeviceOrientationEvent', ifaces.DeviceOrientationEvent);
  vi.stubGlobal('navigator', o.navigator ?? {});
  vi.stubGlobal('document', doc);
  if (o.sensorClass) vi.stubGlobal('RelativeOrientationSensor', o.sensorClass);
  return { win, ifaces, doc };
}

const FULL_MOTION = {
  acceleration: { x: 0.1, y: -0.2, z: 0.3 },
  accelerationIncludingGravity: { x: 1.5, y: 9.6, z: -0.4 },
  rotationRate: { alpha: 10, beta: -20, gamma: 30 },
  interval: 16,
};

// ───────────────────────────── detectCapabilities ─────────────────────────────

describe('detectCapabilities', () => {
  it('plain Node: nothing available, no permission API (and a schema-valid object)', () => {
    vi.stubGlobal('navigator', {});
    const caps = detectCapabilities();
    expect(caps).toEqual({
      secure_context: false,
      device_motion: false,
      device_orientation: false,
      permission_api: false,
      permission_state: 'not_required',
      generic_relative_orientation: false,
      wake_lock: false,
      vibrate: false,
      share_files: false,
      measured: null,
    });
    expectValidDef('SensorCapabilities', caps);
  });

  it('iPhone Safari: permission API, wake lock, file sharing, no vibration, no Generic Sensors', () => {
    const canShare = vi.fn((d: { files: File[] }) => d.files.every((f) => f.type === 'text/plain'));
    installBrowser({
      webkit: true,
      motionPermission: 'granted',
      orientationPermission: 'granted',
      navigator: { wakeLock: { request: () => Promise.resolve({}) }, canShare, share: () => Promise.resolve() },
    });
    const caps = detectCapabilities();
    expect(caps).toMatchObject({
      secure_context: true,
      device_motion: true,
      device_orientation: true,
      permission_api: true,
      permission_state: 'unknown', // not decided until requestMotionPermission() runs
      generic_relative_orientation: false,
      wake_lock: true,
      vibrate: false,
      share_files: true,
      measured: null,
    });
    // probed with a tiny text/plain .txt File (Chrome's share allowlist rejects JSON)
    const probe = canShare.mock.calls[0][0].files[0];
    expect(probe.type).toBe('text/plain');
    expect(probe.name.endsWith('.txt')).toBe(true);
    expectValidDef('SensorCapabilities', caps);
  });

  it('Android Chrome 151+: requestPermission exists too → permission_api, plus Generic Sensors and vibration', () => {
    const { Sensor } = fakeOrientationSensorClass();
    installBrowser({
      motionPermission: 'granted',
      orientationPermission: 'granted',
      sensorClass: Sensor,
      navigator: { vibrate: () => true, canShare: () => false, wakeLock: { request: () => Promise.resolve({}) } },
    });
    expect(detectCapabilities()).toMatchObject({
      permission_api: true,
      permission_state: 'unknown',
      generic_relative_orientation: true,
      vibrate: true,
      share_files: false,
      wake_lock: true,
    });
  });

  it('older Chrome / Firefox Android: no permission API → not_required', () => {
    installBrowser({ navigator: { vibrate: () => true } });
    expect(detectCapabilities()).toMatchObject({ permission_api: false, permission_state: 'not_required', device_motion: true });
  });

  it('insecure context and throwing probes never break detection', () => {
    const nav: Record<string, unknown> = {
      canShare: () => {
        throw new TypeError('not allowed');
      },
    };
    Object.defineProperty(nav, 'wakeLock', {
      get() {
        throw new Error('SecurityError');
      },
    });
    installBrowser({ secure: false, navigator: nav });
    const caps = detectCapabilities();
    expect(caps.secure_context).toBe(false);
    expect(caps.share_files).toBe(false);
    expect(caps.wake_lock).toBe(false);
  });

  it('falls back to location when isSecureContext is missing', () => {
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('location', { protocol: 'https:', hostname: 'example.org' });
    expect(detectCapabilities().secure_context).toBe(true);
    vi.stubGlobal('location', { protocol: 'http:', hostname: 'localhost' });
    expect(detectCapabilities().secure_context).toBe(true);
    vi.stubGlobal('location', { protocol: 'http:', hostname: '192.168.1.20' });
    expect(detectCapabilities().secure_context).toBe(false);
  });
});

// ───────────────────────────── requestMotionPermission ─────────────────────────────

describe('requestMotionPermission', () => {
  it('no permission API (Android Chrome < 151, Firefox, desktop) → not_required', async () => {
    installBrowser();
    await expect(requestMotionPermission()).resolves.toBe('not_required');
  });

  it('no motion interfaces at all → not_required, never throws', async () => {
    await expect(requestMotionPermission()).resolves.toBe('not_required');
  });

  it('iOS: both prompts are started SYNCHRONOUSLY, as methods of their interfaces', async () => {
    const { ifaces } = installBrowser({ webkit: true, motionPermission: 'granted', orientationPermission: 'granted' });
    const p = requestMotionPermission();
    // Both calls already happened before anything was awaited (WebKit's user-gesture rule).
    expect(ifaces.DeviceMotionEvent.requestPermission).toHaveBeenCalledTimes(1);
    expect(ifaces.DeviceOrientationEvent.requestPermission).toHaveBeenCalledTimes(1);
    await expect(p).resolves.toBe('granted'); // the fakes throw "Illegal invocation" when unbound
  });

  it('Chrome 151-style (both functions resolve granted) → granted', async () => {
    installBrowser({ motionPermission: 'granted', orientationPermission: 'granted' });
    await expect(requestMotionPermission()).resolves.toBe('granted');
  });

  it('only DeviceMotionEvent has the function → only that result counts', async () => {
    const { ifaces } = installBrowser({ motionPermission: 'granted' });
    await expect(requestMotionPermission()).resolves.toBe('granted');
    expect(ifaces.DeviceMotionEvent.requestPermission).toHaveBeenCalledTimes(1);
  });

  it.each<[PermissionBehavior, PermissionBehavior, string]>([
    ['denied', 'denied', 'denied'],
    ['granted', 'denied', 'denied'],
    ['denied', 'reject_not_allowed', 'denied'],
    ['reject_not_allowed', 'reject_not_allowed', 'unknown'],
    ['granted', 'reject_not_allowed', 'unknown'],
    ['throw', 'throw', 'unknown'],
    ['throw', 'granted', 'unknown'],
  ])('motion %s + orientation %s → %s', async (m, o, expected) => {
    installBrowser({ webkit: true, motionPermission: m, orientationPermission: o });
    let p!: Promise<string>;
    expect(() => {
      p = requestMotionPermission();
    }).not.toThrow();
    await expect(p).resolves.toBe(expected);
  });

  it('a non-standard answer ("prompt") is not treated as granted', async () => {
    installBrowser();
    vi.stubGlobal('DeviceMotionEvent', { requestPermission: () => Promise.resolve('prompt') });
    await expect(requestMotionPermission()).resolves.toBe('unknown');
  });
});

// ───────────────────────────── browser sensor source ─────────────────────────────

describe('createBrowserSensorSource: samples', () => {
  it('converts devicemotion events (t = performance.now() in the handler, ts = event.timeStamp)', async () => {
    const { win } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    expect(src.kind).toBe('browser');
    await src.start(c.listener);
    const now = vi.spyOn(performance, 'now').mockReturnValueOnce(1234.5);
    win.dispatchEvent(motionEvent({ ...FULL_MOTION, timeStamp: 987.25 }));
    expect(now).toHaveBeenCalledTimes(1);
    expect(c.motion).toEqual([
      {
        t: 1234.5,
        ts: 987.25,
        acc: [0.1, -0.2, 0.3],
        accg: [1.5, 9.6, -0.4],
        rr: { alpha: 10, beta: -20, gamma: 30 }, // copied field by field, never remapped
        interval: 16,
      },
    ]);
    src.stop();
  });

  it('turns every missing or broken field into null', async () => {
    const { win } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    // iOS without a gyro / desktop Chrome: objects present, fields null
    win.dispatchEvent(
      motionEvent({
        acceleration: { x: null, y: null, z: null },
        accelerationIncludingGravity: { x: null, y: null, z: null },
        rotationRate: { alpha: null, beta: null, gamma: null },
        interval: 16,
        timeStamp: Number.NaN,
      }),
    );
    // objects missing entirely, interval missing
    win.dispatchEvent(motionEvent({ acceleration: null, rotationRate: null }));
    // partially filled / wrong types / non-finite
    win.dispatchEvent(
      motionEvent({
        acceleration: { x: 1, y: Number.POSITIVE_INFINITY, z: 0 },
        accelerationIncludingGravity: { x: 1, y: '9.8', z: 0 },
        rotationRate: { alpha: 1, beta: null, gamma: 3 },
        interval: 'fast',
      }),
    );
    expect(c.motion).toHaveLength(3);
    for (const s of c.motion) {
      expect(s.acc).toBeNull();
      expect(s.accg).toBeNull();
      expect(s.rr).toBeNull();
    }
    expect(c.motion[0].interval).toBe(16);
    expect(c.motion[0].ts).toBeNull();
    expect(c.motion[1].interval).toBeNull();
    expect(c.motion[2].interval).toBeNull();
    expect(typeof c.motion[1].ts).toBe('number'); // Node's own Event.timeStamp
    // a platform without linear acceleration (old Android): only acc is null
    win.dispatchEvent(motionEvent({ ...FULL_MOTION, acceleration: null }));
    expect(c.motion[3].acc).toBeNull();
    expect(c.motion[3].accg).toEqual([1.5, 9.6, -0.4]);
    src.stop();
  });

  it('converts deviceorientation events, including iOS webkitCompassHeading', async () => {
    const { win } = installBrowser({ webkit: true });
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    vi.spyOn(performance, 'now').mockReturnValueOnce(50).mockReturnValueOnce(66);
    win.dispatchEvent(
      orientationEvent({ alpha: 120.5, beta: 80.25, gamma: -3.5, absolute: false, webkitCompassHeading: 271.5, timeStamp: 40 }),
    );
    win.dispatchEvent(orientationEvent({ alpha: null, beta: undefined, absolute: 'yes', timeStamp: 56 }));
    expect(c.orientation).toEqual([
      { t: 50, ts: 40, alpha: 120.5, beta: 80.25, gamma: -3.5, absolute: false, compass: 271.5 },
      { t: 66, ts: 56, alpha: null, beta: null, gamma: null, absolute: null, compass: null },
    ]);
    src.stop();
  });

  it('a listener may call stop() from inside onMotion', async () => {
    const { win } = installBrowser();
    let count = 0;
    const src: SensorSource = createBrowserSensorSource({ quaternion: false });
    await src.start({
      onMotion: () => {
        count++;
        src.stop();
      },
      onOrientation: () => undefined,
    });
    win.dispatchEvent(motionEvent(FULL_MOTION));
    win.dispatchEvent(motionEvent(FULL_MOTION));
    expect(count).toBe(1);
    expect(win.listenerCount()).toBe(0);
  });
});

describe('createBrowserSensorSource: lifecycle', () => {
  it('stop() removes every listener and later events are ignored', async () => {
    const { win } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    expect(win.listenerCount('devicemotion')).toBe(1);
    expect(win.listenerCount('deviceorientation')).toBe(1);
    src.stop();
    src.stop(); // idempotent
    expect(win.listenerCount()).toBe(0);
    win.dispatchEvent(motionEvent(FULL_MOTION));
    win.dispatchEvent(orientationEvent({ alpha: 1, beta: 2, gamma: 3 }));
    expect(c.motion).toHaveLength(0);
    expect(c.orientation).toHaveLength(0);
  });

  it('start() twice throws (sources are single-use, also after stop)', async () => {
    installBrowser();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(collector().listener);
    expect(() => src.start(collector().listener)).toThrow(/twice/);
    src.stop();
    expect(() => src.start(collector().listener)).toThrow();
  });

  it('clock: now() is performance.now(), epochAt() maps to the wall clock', () => {
    vi.useFakeTimers({ ...FAKE_TIMERS, now: new Date('2026-10-23T12:00:00Z') });
    installBrowser();
    const src = createBrowserSensorSource({ quaternion: false });
    vi.advanceTimersByTime(250);
    const t = src.clock.now();
    expect(t).toBe(performance.now());
    expect(src.clock.epochAt(t)).toBe(Date.now());
    expect(src.clock.epochAt(t - 100)).toBe(Date.now() - 100);
  });
});

describe('createBrowserSensorSource: no_events watchdog', () => {
  it(`reports no_events once, ${NO_EVENTS_TIMEOUT_MS} ms after start without a devicemotion event`, async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    vi.advanceTimersByTime(NO_EVENTS_TIMEOUT_MS - 1);
    expect(c.errors).toEqual([]);
    vi.advanceTimersByTime(1);
    expect(c.errors).toHaveLength(1);
    expect(c.errors[0].code).toBe('no_events');
    vi.advanceTimersByTime(10_000);
    expect(c.errors).toHaveLength(1);
    src.stop();
  });

  it('a data-carrying event before the deadline cancels it', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { win } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    vi.advanceTimersByTime(100);
    win.dispatchEvent(motionEvent(FULL_MOTION));
    vi.advanceTimersByTime(5000);
    expect(c.errors).toEqual([]);
    src.stop();
  });

  it('events without any values (desktop Chrome without sensors) do not count, but are delivered', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { win } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    win.dispatchEvent(motionEvent({ acceleration: null, accelerationIncludingGravity: null, rotationRate: null }));
    win.dispatchEvent(motionEvent({ acceleration: null, accelerationIncludingGravity: null, rotationRate: null }));
    vi.advanceTimersByTime(NO_EVENTS_TIMEOUT_MS);
    expect(c.motion).toHaveLength(2);
    expect(c.errors).toHaveLength(1);
    expect(c.errors[0].code).toBe('no_events');
    expect(c.errors[0].message).toContain('2 events without values');
    src.stop();
  });

  it('is postponed while the page is hidden (no events are delivered to hidden pages)', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { doc } = installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    doc.setVisibility('hidden');
    vi.advanceTimersByTime(NO_EVENTS_TIMEOUT_MS * 4);
    expect(c.errors).toEqual([]);
    doc.setVisibility('visible');
    vi.advanceTimersByTime(NO_EVENTS_TIMEOUT_MS);
    expect(c.errors.map((e) => e.code)).toEqual(['no_events']);
    src.stop();
  });

  it('stop() cancels the watchdog', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    installBrowser();
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    src.stop();
    vi.advanceTimersByTime(NO_EVENTS_TIMEOUT_MS * 2);
    expect(c.errors).toEqual([]);
  });
});

describe('createBrowserSensorSource: environment errors', () => {
  it('insecure context → insecure_context on a later task (not synchronously inside start)', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { win } = installBrowser({ secure: false });
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    const p = src.start(c.listener);
    expect(c.errors).toEqual([]);
    await p;
    vi.advanceTimersByTime(0);
    expect(c.errors.map((e) => e.code)).toEqual(['insecure_context']);
    expect(win.listenerCount('devicemotion')).toBe(1); // still listening, just in case
    src.stop();
  });

  it('no DeviceMotionEvent → not_supported', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    installBrowser({ noMotionInterface: true });
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: false });
    await src.start(c.listener);
    vi.advanceTimersByTime(0);
    expect(c.errors.map((e) => e.code)).toEqual(['not_supported']);
    src.stop();
  });

  it('no window at all (Node, workers) → not_supported and start() still resolves', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const c = collector();
    const src = createBrowserSensorSource();
    await expect(src.start(c.listener)).resolves.toBeUndefined();
    vi.advanceTimersByTime(0);
    expect(c.errors.map((e) => e.code)).toEqual(['not_supported']);
    src.stop();
  });
});

describe('createBrowserSensorSource: RelativeOrientationSensor quaternions', () => {
  const granted = { accelerometer: 'granted', gyroscope: 'granted' };

  it('runs at 60 Hz in the device frame when both sensor permissions are granted', async () => {
    const { Sensor, instances } = fakeOrientationSensorClass();
    installBrowser({ sensorClass: Sensor, navigator: { permissions: fakePermissionsApi(granted) } });
    const c = collector();
    const src = createBrowserSensorSource({ quaternion: true });
    await src.start(c.listener);
    expect(instances).toHaveLength(1);
    expect(instances[0].options).toEqual({ frequency: 60, referenceFrame: 'device' });
    expect(instances[0].started).toBe(1);
    vi.spyOn(performance, 'now').mockReturnValueOnce(777);
    instances[0].emitReading([0.1, 0.2, 0.3, 0.927]);
    expect(c.quaternion).toEqual([{ t: 777, q: [0.1, 0.2, 0.3, 0.927], sensor: 'RelativeOrientationSensor' }]);
    instances[0].emitReading(null); // no reading yet
    instances[0].emitReading([0, 0, Number.NaN, 1]);
    expect(c.quaternion).toHaveLength(1);
    src.stop();
    expect(instances[0].stopped).toBe(1);
    expect(instances[0].listenerCount()).toBe(0);
  });

  it('honours frequencyHz and defaults the quaternion stream to on', async () => {
    const { Sensor, instances } = fakeOrientationSensorClass();
    installBrowser({ sensorClass: Sensor, navigator: { permissions: fakePermissionsApi(granted) } });
    const src = createBrowserSensorSource({ frequencyHz: 30 });
    await src.start(collector().listener);
    expect(instances[0].options.frequency).toBe(30);
    src.stop();
  });

  it("a sensor 'error' stops only the quaternion stream (no onError, motion keeps flowing)", async () => {
    const { Sensor, instances } = fakeOrientationSensorClass();
    const { win } = installBrowser({ sensorClass: Sensor, navigator: { permissions: fakePermissionsApi(granted) } });
    const c = collector();
    const src = createBrowserSensorSource();
    await src.start(c.listener);
    instances[0].emitReading([0, 0, 0, 1]);
    instances[0].emitError();
    expect(instances[0].stopped).toBe(1);
    expect(instances[0].listenerCount()).toBe(0);
    instances[0].emitReading([0, 0, 0.1, 0.995]);
    expect(c.quaternion).toHaveLength(1);
    win.dispatchEvent(motionEvent(FULL_MOTION));
    expect(c.motion).toHaveLength(1);
    expect(c.errors).toEqual([]);
    src.stop();
    expect(instances[0].stopped).toBe(1); // not stopped twice
  });

  it.each<[string, Record<string, unknown>]>([
    ['no Permissions API', {}],
    ['permission names unknown (query rejects)', { permissions: fakePermissionsApi({}) }],
    ['accelerometer denied', { permissions: fakePermissionsApi({ accelerometer: 'denied', gyroscope: 'granted' }) }],
    ['gyroscope prompt', { permissions: fakePermissionsApi({ accelerometer: 'granted', gyroscope: 'prompt' }) }],
  ])('never assumes Generic Sensor permission: %s → no sensor', async (_label, nav) => {
    const { Sensor, instances } = fakeOrientationSensorClass();
    const { win } = installBrowser({ sensorClass: Sensor, navigator: nav });
    const c = collector();
    const src = createBrowserSensorSource();
    await src.start(c.listener);
    expect(instances).toHaveLength(0);
    win.dispatchEvent(motionEvent(FULL_MOTION));
    expect(c.motion).toHaveLength(1);
    src.stop();
  });

  it('a hanging permission query gives up after 500 ms', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { Sensor, instances } = fakeOrientationSensorClass();
    installBrowser({ sensorClass: Sensor, navigator: { permissions: { query: () => new Promise(() => undefined) } } });
    const src = createBrowserSensorSource();
    let resolved = false;
    void src.start(collector().listener).then(() => {
      resolved = true;
    });
    await vi.advanceTimersByTimeAsync(499);
    expect(resolved).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(resolved).toBe(true);
    expect(instances).toHaveLength(0);
    src.stop();
  });

  it('quaternion: false never touches the sensor; a throwing constructor is harmless', async () => {
    const off = fakeOrientationSensorClass();
    installBrowser({ sensorClass: off.Sensor, navigator: { permissions: fakePermissionsApi(granted) } });
    const a = createBrowserSensorSource({ quaternion: false });
    await a.start(collector().listener);
    expect(off.instances).toHaveLength(0);
    a.stop();

    const blocked = fakeOrientationSensorClass({ throwOnConstruct: true });
    vi.stubGlobal('RelativeOrientationSensor', blocked.Sensor);
    const c = collector();
    const b = createBrowserSensorSource();
    await expect(b.start(c.listener)).resolves.toBeUndefined();
    expect(c.errors).toEqual([]);
    b.stop();
  });

  it('stop() during the permission query → the sensor is never created', async () => {
    const { Sensor, instances } = fakeOrientationSensorClass();
    installBrowser({ sensorClass: Sensor, navigator: { permissions: fakePermissionsApi(granted) } });
    const src = createBrowserSensorSource();
    const p = src.start(collector().listener);
    src.stop();
    await p;
    expect(instances).toHaveLength(0);
  });
});

// ───────────────────────────── rates ─────────────────────────────

describe('rateSummary', () => {
  it('rate from the span, median and p95 (linear interpolation) of the spacing', () => {
    expect(rateSummary([0, 10, 20, 40])).toEqual({ n: 4, hz: 75, dt_median: 10, dt_p95: 19 });
    expect(rateSummary([5])).toEqual({ n: 1, hz: null, dt_median: null, dt_p95: null });
    expect(rateSummary([])).toEqual({ n: 0, hz: null, dt_median: null, dt_p95: null });
    expect(rateSummary([3, 3]).hz).toBeNull();
  });
});

/** Offline-style source: delivers every sample synchronously inside start() until stopped. */
function syncSource(opts: { hz: number; maxMs: number; acc?: boolean; rr?: boolean; startTime?: number }) {
  let virtualNow = opts.startTime ?? 1000;
  let stopped = false;
  const state = { delivered: 0, stopCalls: 0 };
  const source: SensorSource = {
    kind: 'simulated',
    clock: { now: () => virtualNow, epochAt: (t) => 1.7e12 + t },
    start(listener) {
      const dt = 1000 / opts.hz;
      const end = virtualNow + opts.maxMs;
      while (!stopped && virtualNow <= end) {
        state.delivered++;
        listener.onMotion({
          t: virtualNow,
          ts: virtualNow,
          acc: opts.acc === false ? null : [0, 0, 0],
          accg: [0, 9.8, 0],
          rr: opts.rr === false ? null : { alpha: 0, beta: 0, gamma: 0 },
          interval: 0.016666666666666666,
        });
        if (state.delivered % 2 === 0) listener.onOrientation({ t: virtualNow, ts: null, alpha: 0, beta: 90, gamma: 0, absolute: false, compass: null });
        virtualNow += dt;
      }
      return Promise.resolve();
    },
    stop() {
      stopped = true;
      state.stopCalls++;
    },
  };
  return { source, state };
}

describe('measureRates', () => {
  it('stops an offline (synchronous) source from inside the listener once durationMs elapsed', async () => {
    const { source, state } = syncSource({ hz: 60, maxMs: 60_000 });
    const r = await measureRates(source, 1000);
    expect(state.delivered).toBe(61); // t0 … t0 + 1000 ms, not the whole minute
    expect(state.stopCalls).toBeGreaterThanOrEqual(1);
    expect(r.motion_hz!).toBeCloseTo(60, 6);
    expect(r.orientation_hz!).toBeCloseTo(30, 6);
    expect(r.quaternion_hz).toBeNull();
    expect(r.motion_dt_ms_median!).toBeCloseTo(16.667, 2);
    expect(r.motion_dt_ms_p95!).toBeCloseTo(16.667, 2);
    expect(r.duration_ms).toBeCloseTo(1000, 0);
    expect(r).toMatchObject({ acc_available: true, accg_available: true, rotation_available: true });
    expect(r.interval_raw).toBeCloseTo(0.0167, 3); // iOS reports SECONDS: stored exactly as given
    expectValidDef('MeasuredRates', r);
  });

  it('flags missing streams', async () => {
    const { source } = syncSource({ hz: 50, maxMs: 60_000, acc: false, rr: false });
    const r = await measureRates(source, 500);
    expect(r).toMatchObject({ acc_available: false, accg_available: true, rotation_available: false });
    expect(r.motion_hz!).toBeCloseTo(50, 6);
  });

  it('measures the real browser source with timers and detaches it afterwards', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const { win } = installBrowser();
    const src = createBrowserSensorSource({ quaternion: false });
    const ticker = setInterval(() => win.dispatchEvent(motionEvent({ ...FULL_MOTION, interval: 16 })), 16);
    const p = measureRates(src, 2000);
    await vi.advanceTimersByTimeAsync(2100);
    const r = await p;
    clearInterval(ticker);
    expect(r.motion_hz!).toBeCloseTo(62.5, 0);
    expect(r.motion_dt_ms_median).toBe(16);
    expect(r.interval_raw).toBe(16);
    expect(r.duration_ms).toBeGreaterThanOrEqual(2000);
    expect(win.listenerCount()).toBe(0);
    expectValidDef('MeasuredRates', r);
  });

  it('no events at all → resolves after durationMs with null rates', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    installBrowser();
    const p = measureRates(createBrowserSensorSource({ quaternion: false }), 1500);
    await vi.advanceTimersByTimeAsync(1500);
    const r = await p;
    expect(r).toMatchObject({
      motion_hz: null,
      orientation_hz: null,
      quaternion_hz: null,
      motion_dt_ms_median: null,
      motion_dt_ms_p95: null,
      acc_available: false,
      accg_available: false,
      rotation_available: false,
      interval_raw: null,
    });
    expectValidDef('MeasuredRates', r);
  });

  it('propagates a failing start() after stopping the source', async () => {
    const stop = vi.fn();
    const bad: SensorSource = {
      kind: 'simulated',
      clock: { now: () => 0, epochAt: (t) => t },
      start: () => Promise.reject(new Error('cannot start')),
      stop,
    };
    await expect(measureRates(bad, 100)).rejects.toThrow('cannot start');
    expect(stop).toHaveBeenCalled();
    const throwing: SensorSource = { ...bad, start: () => { throw new Error('sync'); } };
    await expect(measureRates(throwing, 100)).rejects.toThrow('sync');
  });
});

// ───────────────────────────── device info ─────────────────────────────

const UA = {
  iphoneSafari17:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Mobile/15E148 Safari/604.1',
  iphoneSafari18:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1',
  iphoneSafari26:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1',
  chromeIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/126.0.6478.153 Mobile/15E148 Safari/604.1',
  firefoxIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/128.0 Mobile/15E148 Safari/605.1.15',
  edgeIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 EdgiOS/126.0.2592.80 Mobile/15E148 Safari/605.1.15',
  ipadDesktop:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
  androidChromeReduced:
    'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
  samsungInternet:
    'Mozilla/5.0 (Linux; Android 14; SAMSUNG SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/26.0 Chrome/122.0.0.0 Mobile Safari/537.36',
  samsungInternetReduced:
    'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/27.0 Chrome/125.0.0.0 Mobile Safari/537.36',
  firefoxAndroid: 'Mozilla/5.0 (Android 14; Mobile; rv:131.0) Gecko/131.0 Firefox/131.0',
  edgeAndroid:
    'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36 EdgA/131.0.2903.70',
  instagramIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Instagram 339.0.3.12.91 (iPhone15,2; iOS 17_5_1; en_US; en; scale=3.00; 1179x2556; 619446361; IABMV/1)',
  facebookIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 [FBAN/FBIOS;FBAV/470.0.0.40.97;FBBV/614286620;FBDV/iPhone14,5;FBMD/iPhone;FBSN/iOS;FBSV/17.5;FBSS/3;FBID/phone;FBLC/en_US;FBOP/5;FBRV/616254588]',
  instagramAndroid:
    'Mozilla/5.0 (Linux; Android 14; SM-S918B Build/UP1A.231005.007; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/127.0.6533.103 Mobile Safari/537.36 Instagram 343.0.0.33.101 Android (34/14; 480dpi; 1080x2340; samsung; SM-S918B; dm3q; qcom; en_US; 626394000)',
  facebookAndroid:
    'Mozilla/5.0 (Linux; Android 13; Pixel 7 Build/TQ3A.230901.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/116.0.0.0 Mobile Safari/537.36 [FB_IAB/FB4A;FBAV/432.0.0.29.102;]',
  androidWebView:
    'Mozilla/5.0 (Linux; Android 13; SM-A536B Build/TP1A.220624.014; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/119.0.6045.163 Mobile Safari/537.36',
  googleAppIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) GSA/328.0.658461140 Mobile/15E148 Safari/604.1',
  tiktokIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 musical_ly_35.3.0 JsSdk/2.0 NetType/WIFI Channel/App Store ByteLocale/en Region/US',
  snapchatIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Snapchat/13.0.0.42 (like Safari/8618.2.12.10.5, panda)',
  lineIos:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Safari Line/14.10.0',
  bareWkWebView: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
  desktopChrome:
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
  desktopEdge:
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0',
  desktopFirefox: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
  desktopSafari:
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
};

interface UaCase {
  name: string;
  ua: string;
  /** DeviceOrientationEvent.prototype has webkitCompassHeading (true for every iOS engine) */
  webkit: boolean;
  touch?: number;
  uaData?: Record<string, unknown> | null;
  brave?: boolean;
  standalone?: boolean;
  expected: {
    platform: string;
    browser: string;
    os_version: string | null;
    model_hint: string | null;
    engine: string;
    in_app_browser: string | null;
  };
}

const UA_CASES: UaCase[] = [
  {
    name: 'iPhone Safari 17',
    ua: UA.iphoneSafari17,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Safari 17.6', os_version: '17.6.1', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'iPhone Safari 18',
    ua: UA.iphoneSafari18,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Safari 18.1', os_version: '18.1', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'iPhone Safari 26 (OS version frozen at 18_6 in the UA → taken from Version/)',
    ua: UA.iphoneSafari26,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Safari 26.0', os_version: '26.0', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'Chrome on iOS (WebKit)',
    ua: UA.chromeIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Chrome iOS 126.0.6478.153', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'Firefox on iOS (WebKit, not Gecko)',
    ua: UA.firefoxIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Firefox iOS 128.0', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'Edge on iOS',
    ua: UA.edgeIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Edge iOS 126.0.2592.80', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'iPadOS Safari in desktop mode (claims to be a Mac, has touch)',
    ua: UA.ipadDesktop,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Safari 17.5', os_version: '17.5', model_hint: 'iPad', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'Android Chrome, reduced UA, no UA-CH (version and model unknown, not "10"/"K")',
    ua: UA.androidChromeReduced,
    webkit: false,
    expected: { platform: 'android', browser: 'Chrome 131.0.0.0', os_version: null, model_hint: null, engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Android Chrome, reduced UA + UA Client Hints',
    ua: UA.androidChromeReduced,
    webkit: false,
    uaData: { model: 'Pixel 8', platformVersion: '14.0.0' },
    expected: { platform: 'android', browser: 'Chrome 131.0.0.0', os_version: '14.0.0', model_hint: 'Pixel 8', engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Brave Android (plain Chrome UA + navigator.brave; delivers no motion events)',
    ua: UA.androidChromeReduced,
    webkit: false,
    brave: true,
    uaData: { model: 'Pixel 7', platformVersion: '15.0.0' },
    expected: { platform: 'android', browser: 'Brave 131.0.0.0', os_version: '15.0.0', model_hint: 'Pixel 7', engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Samsung Internet (full UA)',
    ua: UA.samsungInternet,
    webkit: false,
    expected: { platform: 'android', browser: 'Samsung Internet 26.0', os_version: '14', model_hint: 'SM-S918B', engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Samsung Internet (reduced UA)',
    ua: UA.samsungInternetReduced,
    webkit: false,
    expected: { platform: 'android', browser: 'Samsung Internet 27.0', os_version: null, model_hint: null, engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Firefox Android',
    ua: UA.firefoxAndroid,
    webkit: false,
    expected: { platform: 'android', browser: 'Firefox 131.0', os_version: '14', model_hint: null, engine: 'gecko', in_app_browser: null },
  },
  {
    name: 'Edge Android',
    ua: UA.edgeAndroid,
    webkit: false,
    expected: { platform: 'android', browser: 'Edge Android 131.0.2903.70', os_version: null, model_hint: null, engine: 'blink', in_app_browser: null },
  },
  {
    name: 'Instagram in-app browser, iOS (hardware model in the UA)',
    ua: UA.instagramIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Instagram 339.0.3.12.91', os_version: '17.5.1', model_hint: 'iPhone15,2', engine: 'webkit', in_app_browser: 'instagram' },
  },
  {
    name: 'Facebook in-app browser, iOS',
    ua: UA.facebookIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Facebook 470.0.0.40.97', os_version: '17.5', model_hint: 'iPhone14,5', engine: 'webkit', in_app_browser: 'facebook' },
  },
  {
    name: 'Instagram in-app browser, Android (WebView)',
    ua: UA.instagramAndroid,
    webkit: false,
    expected: { platform: 'android', browser: 'Instagram 343.0.0.33.101', os_version: '14', model_hint: 'SM-S918B', engine: 'blink', in_app_browser: 'instagram' },
  },
  {
    name: 'Facebook in-app browser, Android',
    ua: UA.facebookAndroid,
    webkit: false,
    expected: { platform: 'android', browser: 'Facebook 432.0.0.29.102', os_version: '13', model_hint: 'Pixel 7', engine: 'blink', in_app_browser: 'facebook' },
  },
  {
    name: 'Android WebView (wv token)',
    ua: UA.androidWebView,
    webkit: false,
    expected: { platform: 'android', browser: 'Android WebView 119.0.6045.163', os_version: '13', model_hint: 'SM-A536B', engine: 'blink', in_app_browser: 'android_webview' },
  },
  {
    name: 'Google app (GSA), iOS',
    ua: UA.googleAppIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Google app 328.0.658461140', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: 'google_app' },
  },
  {
    name: 'TikTok in-app browser, iOS',
    ua: UA.tiktokIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'TikTok 35.3.0', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: 'tiktok' },
  },
  {
    name: 'Snapchat in-app browser, iOS',
    ua: UA.snapchatIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'Snapchat 13.0.0.42', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: 'snapchat' },
  },
  {
    name: 'LINE in-app browser, iOS',
    ua: UA.lineIos,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'LINE 14.10.0', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: 'line' },
  },
  {
    name: 'bare WKWebView inside some iOS app',
    ua: UA.bareWkWebView,
    webkit: true,
    touch: 5,
    expected: { platform: 'ios', browser: 'iOS WebView', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: 'ios_webview' },
  },
  {
    name: 'iOS Home Screen web app (same bare UA, but standalone)',
    ua: UA.bareWkWebView,
    webkit: true,
    touch: 5,
    standalone: true,
    expected: { platform: 'ios', browser: 'Safari', os_version: '17.5', model_hint: 'iPhone', engine: 'webkit', in_app_browser: null },
  },
  {
    name: 'desktop Chrome (Windows) with UA-CH',
    ua: UA.desktopChrome,
    webkit: false,
    uaData: { model: '', platformVersion: '15.0.0' },
    expected: { platform: 'other', browser: 'Chrome 131.0.0.0', os_version: '15.0.0', model_hint: null, engine: 'blink', in_app_browser: null },
  },
  {
    name: 'desktop Edge',
    ua: UA.desktopEdge,
    webkit: false,
    expected: { platform: 'other', browser: 'Edge 131.0.0.0', os_version: '10.0', model_hint: null, engine: 'blink', in_app_browser: null },
  },
  {
    name: 'desktop Firefox',
    ua: UA.desktopFirefox,
    webkit: false,
    expected: { platform: 'other', browser: 'Firefox 132.0', os_version: '10.0', model_hint: null, engine: 'gecko', in_app_browser: null },
  },
  {
    name: "desktop Safari (macOS: no webkitCompassHeading → engine 'unknown', no motion sensors)",
    ua: UA.desktopSafari,
    webkit: false,
    touch: 0,
    expected: { platform: 'other', browser: 'Safari 18.1', os_version: '10.15.7', model_hint: null, engine: 'unknown', in_app_browser: null },
  },
];

function installUa(c: Pick<UaCase, 'ua' | 'webkit' | 'touch' | 'uaData' | 'brave' | 'standalone'>, extra: Record<string, unknown> = {}) {
  const nav: Record<string, unknown> = {
    userAgent: c.ua,
    maxTouchPoints: c.touch ?? 0,
    language: 'he-IL',
    hardwareConcurrency: 6,
    ...extra,
  };
  if (c.uaData) {
    const values = c.uaData;
    nav.userAgentData = { getHighEntropyValues: vi.fn(() => Promise.resolve(values)) };
  }
  if (c.brave) nav.brave = { isBrave: () => Promise.resolve(true) };
  if (c.standalone) nav.standalone = true;
  vi.stubGlobal('navigator', nav);
  vi.stubGlobal('DeviceOrientationEvent', fakeMotionInterfaces({ webkit: c.webkit }).DeviceOrientationEvent);
  vi.stubGlobal('screen', { width: 390, height: 844 });
  vi.stubGlobal('devicePixelRatio', 3);
  return nav;
}

describe('getDeviceInfo: user-agent table', () => {
  for (const c of UA_CASES) {
    it(c.name, async () => {
      installUa(c);
      const info = await getDeviceInfo();
      expect({
        platform: info.platform,
        browser: info.browser,
        os_version: info.os_version,
        model_hint: info.model_hint,
        engine: info.engine,
        in_app_browser: info.in_app_browser,
      }).toEqual(c.expected);
      expect(info.user_agent).toBe(c.ua);
      expect(info.standalone).toBe(!!c.standalone);
      expectValidDef('DeviceInfo', info);
    });
  }

  it('parseUserAgent is pure and reports the reduced Android UA', () => {
    expect(parseUserAgent(UA.androidChromeReduced).reduced_ua).toBe(true);
    expect(parseUserAgent(UA.samsungInternet).reduced_ua).toBe(false);
    expect(parseUserAgent('')).toEqual({
      platform: 'other',
      browser: 'unknown',
      os_version: null,
      model_hint: null,
      in_app_browser: null,
      reduced_ua: false,
    });
  });
});

describe('getDeviceInfo: environment', () => {
  it('copies screen, dpr, language, cores, memory, time origin', async () => {
    installUa({ ua: UA.androidChromeReduced, webkit: false }, { deviceMemory: 8 });
    const info = await getDeviceInfo();
    expect(info).toMatchObject({
      screen_w: 390,
      screen_h: 844,
      dpr: 3,
      language: 'he-IL',
      hardware_concurrency: 6,
      device_memory_gb: 8,
      standalone: false,
    });
    expect(info.time_origin_ms).toBe(performance.timeOrigin);
  });

  it('standalone from the display-mode media query', async () => {
    installUa({ ua: UA.androidChromeReduced, webkit: false });
    vi.stubGlobal('matchMedia', (q: string) => ({ matches: q === '(display-mode: standalone)' }));
    expect((await getDeviceInfo()).standalone).toBe(true);
    vi.stubGlobal('matchMedia', () => {
      throw new SyntaxError('unknown media feature');
    });
    expect((await getDeviceInfo()).standalone).toBe(false);
  });

  it('UA-CH that never answers is abandoned after 500 ms', async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    installUa({ ua: UA.androidChromeReduced, webkit: false }, {
      userAgentData: { getHighEntropyValues: () => new Promise(() => undefined) },
    });
    let info: Awaited<ReturnType<typeof getDeviceInfo>> | null = null;
    void getDeviceInfo().then((i) => {
      info = i;
    });
    await vi.advanceTimersByTimeAsync(499);
    expect(info).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    expect(info).not.toBeNull();
    expect(info!.model_hint).toBeNull();
    expect(info!.engine).toBe('blink');
  });

  it('a rejecting or empty UA-CH answer falls back to the UA', async () => {
    installUa({ ua: UA.samsungInternet, webkit: false }, {
      userAgentData: { getHighEntropyValues: () => Promise.reject(new DOMException('denied', 'NotAllowedError')) },
    });
    expect((await getDeviceInfo()).model_hint).toBe('SM-S918B');
    installUa({ ua: UA.samsungInternet, webkit: false }, {
      userAgentData: { getHighEntropyValues: () => Promise.resolve({ model: '  ', platformVersion: '' }) },
    });
    const info = await getDeviceInfo();
    expect(info.model_hint).toBe('SM-S918B');
    expect(info.os_version).toBe('14');
  });

  it('never throws in a bare runtime and still produces a schema-valid record', async () => {
    vi.stubGlobal('navigator', undefined);
    const info = await getDeviceInfo();
    expect(info).toMatchObject({ user_agent: '', platform: 'other', browser: 'unknown', screen_w: 0, screen_h: 0, dpr: 1, language: '' });
    expectValidDef('DeviceInfo', info);
  });
});

// ───────────────────────────── screen orientation ─────────────────────────────

describe('getScreenOrientation', () => {
  it('reads screen.orientation', () => {
    vi.stubGlobal('screen', { orientation: { type: 'landscape-primary', angle: 90 } });
    expect(getScreenOrientation()).toEqual({ type: 'landscape-primary', angle: 90 });
  });

  it.each<[number, string, number]>([
    [0, 'portrait-primary', 0],
    [90, 'landscape-primary', 90],
    [-90, 'landscape-secondary', 270],
    [180, 'portrait-secondary', 180],
  ])('falls back to window.orientation (old iOS): %i → %s / %i', (legacy, type, angle) => {
    vi.stubGlobal('screen', { width: 390, height: 844 });
    vi.stubGlobal('orientation', legacy);
    expect(getScreenOrientation()).toEqual({ type, angle });
  });

  it('nothing available or a throwing getter → nulls', () => {
    expect(getScreenOrientation()).toEqual({ type: null, angle: null });
    const scr = {};
    Object.defineProperty(scr, 'orientation', {
      get() {
        throw new Error('nope');
      },
    });
    vi.stubGlobal('screen', scr);
    expect(getScreenOrientation()).toEqual({ type: null, angle: null });
  });
});
