/**
 * Real-device sensor access (browser APIs). Everything that touches window/navigator lives here.
 *
 * - detectCapabilities(): feature detection only (no prompts). permission_state is 'unknown' until
 *   requestMotionPermission() has run; 'not_required' where no permission API exists.
 * - requestMotionPermission(): MUST be called synchronously from a user gesture (click/touchend), as
 *   the FIRST statement of the handler: no await (IndexedDB, fetch, AudioContext...) before it —
 *   WebKit only shows the prompt while the gesture is being processed. Call
 *   DeviceMotionEvent.requestPermission?.() AND DeviceOrientationEvent.requestPermission?.()
 *   synchronously (start both promises before awaiting either) on EVERY browser where the function
 *   exists: iOS 13+ and also Chrome 151+ (Android/desktop; currently resolves 'granted', Chrome plans
 *   Ask-by-default). Both granted → 'granted'; any 'denied' → 'denied'; NotAllowedError (no gesture)
 *   → 'unknown'. No function → 'not_required'. Facts: iOS caches the decision per origin IN MEMORY
 *   for the app process (survives reloads, not a Safari force-quit), a Cancel is just as sticky, and
 *   an insecure context or a third-party iframe silently resolves 'denied' without a prompt.
 *   The grant must be requested again on every page load (cheap: resolves instantly once decided).
 *   Generic Sensor permissions (accelerometer/gyroscope) are queried in try/catch, never assumed.
 * - Engine detection (getDeviceInfo().engine): 'webkit' iff 'webkitCompassHeading' in
 *   DeviceOrientationEvent.prototype (all iOS browsers); never infer iOS from requestPermission.
 *   Otherwise 'gecko' (Firefox UA), 'blink' (UA-CH or a Chrome/Chromium token), else 'unknown' —
 *   so desktop Safari (no webkitCompassHeading, no motion sensors) reports 'unknown'.
 *   In-app browsers: Instagram, FBAN/FBAV (Facebook), TikTok/musical_ly/BytedanceWebview, Line,
 *   Snapchat, GSA (Google app), and the Android WebView 'wv' token → in_app_browser. Also
 *   LinkedIn/WeChat/Pinterest/WhatsApp, and 'ios_webview' for an iOS UA without a "Safari/" token outside
 *   Home Screen mode (a bare WKWebView inside some app).
 * - createBrowserSensorSource(): listens to 'devicemotion' and 'deviceorientation' (and, when
 *   available and opts.quaternion !== false, a RelativeOrientationSensor at opts.frequencyHz (default
 *   60)). Converts each event into MotionSample/OrientationSample with t = performance.now() taken
 *   at the top of the handler, ts = event.timeStamp. Null-safe: any missing field becomes null
 *   (a vector with any missing/non-finite component becomes null as a whole). iOS: the orientation
 *   sample carries webkitCompassHeading (compass) and webkitCompassAccuracy (compass_accuracy).
 *   Schema 1.2 extra streams, each best effort and only when the listener implements its callback
 *   (so measureRates never starts them) and opts does not switch it off:
 *     · AbsoluteOrientationSensor (magnetometer-fused, Earth-referenced) → onQuaternionAbs, when the
 *       constructor exists and no Permissions API query of accelerometer / gyroscope / magnetometer
 *       says 'denied' (unlike the relative sensor, an unanswered query does not rule it out);
 *     · 'deviceorientationabsolute' events (Chrome; feature-detected with 'ondeviceorientationabsolute'
 *       in window) → onOrientationAbs;
 *     · Magnetometer (µT; exposed only behind a flag in today's browsers; feature-detected, never
 *       prompted, same permission rule as the absolute sensor) → onMagnetometer;
 *     · screen orientation 'change' events (screen.orientation, else window 'orientationchange')
 *       → onScreen with getScreenOrientation() after the change.
 *   An optional stream that fails never stops the others and never raises onError: the source calls
 *   listener.onWarning?.(code) (see SensorWarningListener in trialController.ts) with
 *   'quaternion_unavailable', 'abs_orientation_unavailable' / 'abs_orientation_denied',
 *   'magnetometer_unavailable' / 'magnetometer_denied' (construction or start() threw, or an 'error'
 *   event); the trial controller writes these at most once per session.
 *   stop() removes every listener and stops every Generic Sensor. Calling start twice is an error
 *   (synchronous throw; a source is single-use, also after stop()).
 *   If no devicemotion event arrives within 1500 ms after start, listener.onError({code:'no_events'}).
 *   Only events carrying data count (desktop Chrome fires all-null events when it has no sensor),
 *   and the check is postponed while the page is hidden (browsers deliver no events then).
 *   start() reports 'insecure_context' / 'not_supported' through onError on a later task (never
 *   synchronously inside start) and still attaches its listeners. Quaternion sensor errors only stop
 *   the quaternion stream (it is optional; a missing gyroscope must not abort trials).
 *   start() resolves once the optional Generic Sensors have been set up or ruled out.
 * - readTouch(): clientX/clientY of a DOM Touch plus its schema 1.2 contact details (radius, force,
 *   rotation, viewport-normalised position) for TrialController.addTouch.
 * - measureRates(): runs a source for durationMs and summarizes rates (own rateSummary helper: hz =
 *   (n−1)/(t_last−t_first)·1000, median/p95 of the sample spacing). Stops the source from inside the
 *   listener once durationMs of source time has elapsed, so offline simulated sources work too.
 * - getDeviceInfo(): UA parsing (platform, browser, os_version), navigator.userAgentData
 *   getHighEntropyValues(['model','platformVersion']) when present (model_hint, os_version; 500 ms
 *   timeout), standalone (display-mode: standalone/fullscreen media query or navigator.standalone),
 *   screen, dpr, language, etc.
 * - getScreenOrientation(): screen.orientation.{type,angle} or window.orientation fallback.
 *
 * IMPLEMENTED BY: sensors agent.
 */
import type {
  DeviceInfo,
  Euler,
  MeasuredRates,
  MotionSample,
  OrientationSample,
  PermissionState,
  Platform,
  SensorCapabilities,
  SensorError,
  SensorListener,
  SensorSource,
  SourceClock,
  TouchDetails,
  Vec3,
} from './types';

// ───────────────────────────── loose views of browser globals ─────────────────────────────
// lib.dom lacks several of these APIs (requestPermission, userAgentData, Generic Sensors,
// navigator.standalone/deviceMemory/brave) and any of them may be missing at runtime, so every
// access goes through these optional shapes instead of the DOM typings.

interface UaDataLike {
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>;
}

interface LooseNavigator {
  userAgent?: string;
  language?: string;
  hardwareConcurrency?: number;
  deviceMemory?: number;
  maxTouchPoints?: number;
  standalone?: boolean;
  brave?: unknown;
  userAgentData?: UaDataLike | null;
  wakeLock?: { request?: unknown } | null;
  vibrate?: unknown;
  canShare?: (data: { files: unknown[] }) => boolean;
  permissions?: { query?: (descriptor: { name: string }) => Promise<{ state?: string }> } | null;
}

interface GenericSensorLike extends EventTarget {
  readonly quaternion?: ArrayLike<number> | null;
  /** Magnetometer readings (µT) */
  readonly x?: number | null;
  readonly y?: number | null;
  readonly z?: number | null;
  start(): void;
  stop(): void;
}

type GenericSensorCtor = new (options: { frequency: number; referenceFrame: 'device' | 'screen' }) => GenericSensorLike;

/** screen.orientation (an EventTarget in every engine that has it). */
interface ScreenOrientationLike {
  type?: unknown;
  angle?: unknown;
  addEventListener?: EventTarget['addEventListener'];
  removeEventListener?: EventTarget['removeEventListener'];
}

interface BrowserGlobals {
  window?: EventTarget;
  document?: { visibilityState?: string; hidden?: boolean };
  navigator?: LooseNavigator;
  screen?: { width?: number; height?: number; orientation?: ScreenOrientationLike | null };
  location?: { protocol?: string; hostname?: string };
  isSecureContext?: boolean;
  devicePixelRatio?: number;
  innerWidth?: number;
  innerHeight?: number;
  /** deprecated window.orientation (iOS < 16.4 has no screen.orientation) */
  orientation?: unknown;
  matchMedia?: (query: string) => { matches: boolean };
  DeviceMotionEvent?: unknown;
  DeviceOrientationEvent?: unknown;
  RelativeOrientationSensor?: unknown;
  AbsoluteOrientationSensor?: unknown;
  Magnetometer?: unknown;
  File?: unknown;
  performance?: { now(): number; timeOrigin?: number };
}

function G(): BrowserGlobals {
  return globalThis as unknown as BrowserGlobals;
}

function finiteOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function perfNow(): number {
  const p = G().performance;
  return p && typeof p.now === 'function' ? p.now() : Date.now();
}

function timeOrigin(): number {
  const p = G().performance;
  const origin = p ? finiteOrNull(p.timeOrigin) : null;
  return origin ?? Date.now() - perfNow();
}

function isDocumentHidden(): boolean {
  const d = G().document;
  return !!d && (d.visibilityState === 'hidden' || d.hidden === true);
}

function isSecure(): boolean {
  const g = G();
  if (typeof g.isSecureContext === 'boolean') return g.isSecureContext;
  // Engines older than isSecureContext: https or a loopback host is what the motion APIs accept.
  const loc = g.location;
  if (!loc) return false;
  return loc.protocol === 'https:' || loc.hostname === 'localhost' || loc.hostname === '127.0.0.1';
}

/** Resolves with `fallback` if `p` has not settled after `ms` (the timer is always cleared). */
function withTimeout<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<T>((resolve) => {
    timer = setTimeout(() => resolve(fallback), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer));
}

/** DeviceMotionEvent.requestPermission-style static method, bound to its interface object. */
function permissionRequester(iface: unknown): (() => unknown) | null {
  if (iface === null || (typeof iface !== 'function' && typeof iface !== 'object')) return null;
  const fn = (iface as { requestPermission?: unknown }).requestPermission;
  return typeof fn === 'function' ? () => (fn as () => unknown).call(iface) : null;
}

function hasWebkitCompass(): boolean {
  try {
    const iface = G().DeviceOrientationEvent as { prototype?: unknown } | undefined;
    const proto = iface?.prototype;
    return typeof proto === 'object' && proto !== null && 'webkitCompassHeading' in proto;
  } catch {
    return false;
  }
}

function canShareFiles(nav: LooseNavigator | undefined): boolean {
  try {
    const FileCtor = G().File;
    if (!nav || typeof nav.canShare !== 'function' || typeof FileCtor !== 'function') return false;
    // text/plain: Chrome's share allowlist rejects .json, and that is what the exporter falls back to.
    const probe = new (FileCtor as typeof File)(['nosetouch'], 'probe.txt', { type: 'text/plain' });
    return nav.canShare({ files: [probe] }) === true;
  } catch {
    return false;
  }
}

// ───────────────────────────── capabilities & permission ─────────────────────────────

export function detectCapabilities(): SensorCapabilities {
  const g = G();
  const nav = g.navigator;
  const permission_api = permissionRequester(g.DeviceMotionEvent) !== null;
  let wake_lock = false;
  try {
    wake_lock = !!nav?.wakeLock && typeof nav.wakeLock.request === 'function';
  } catch {
    wake_lock = false;
  }
  return {
    secure_context: isSecure(),
    device_motion: g.DeviceMotionEvent !== undefined && g.DeviceMotionEvent !== null,
    device_orientation: g.DeviceOrientationEvent !== undefined && g.DeviceOrientationEvent !== null,
    permission_api,
    permission_state: permission_api ? 'unknown' : 'not_required',
    generic_relative_orientation: typeof g.RelativeOrientationSensor === 'function',
    wake_lock,
    vibrate: typeof nav?.vibrate === 'function',
    share_files: canShareFiles(nav),
    measured: null,
  };
}

function permissionFromResults(results: PromiseSettledResult<unknown>[]): PermissionState {
  let denied = false;
  let allGranted = true;
  for (const r of results) {
    if (r.status === 'fulfilled' && r.value === 'granted') continue;
    allGranted = false;
    if (r.status === 'fulfilled' && r.value === 'denied') denied = true;
    // A rejection is NotAllowedError (no user gesture) or an engine quirk: the decision is still open.
  }
  if (denied) return 'denied';
  return allGranted ? 'granted' : 'unknown';
}

export function requestMotionPermission(): Promise<PermissionState> {
  // Everything up to both requestPermission() calls is synchronous on purpose: WebKit shows the
  // prompt only while the user gesture is being processed. Never insert an await above them.
  const g = G();
  const requesters = [permissionRequester(g.DeviceMotionEvent), permissionRequester(g.DeviceOrientationEvent)];
  const pending: Promise<unknown>[] = [];
  for (const request of requesters) {
    if (!request) continue;
    try {
      pending.push(Promise.resolve(request()));
    } catch (e) {
      pending.push(Promise.reject(e));
    }
  }
  if (pending.length === 0) return Promise.resolve('not_required');
  return Promise.allSettled(pending).then(permissionFromResults, () => 'unknown' as const);
}

/** Generic Sensor permissions are queried, never assumed (an unknown name rejects → not allowed). */
async function genericSensorsGranted(nav: LooseNavigator | undefined): Promise<boolean> {
  const perms = nav?.permissions;
  if (!perms || typeof perms.query !== 'function') return false;
  try {
    const states = await withTimeout(
      Promise.all(['accelerometer', 'gyroscope'].map((name) => perms.query!({ name }).then((s) => s?.state))),
      500,
      null,
    );
    return states !== null && states.every((s) => s === 'granted');
  } catch {
    return false;
  }
}

/**
 * true when a Permissions API query says 'denied' for any of `names` (each query on its own: a
 * rejected, unanswered (500 ms) or missing query is not a denial).
 */
async function anyPermissionDenied(nav: LooseNavigator | undefined, names: readonly string[]): Promise<boolean> {
  const perms = nav?.permissions;
  if (!perms || typeof perms.query !== 'function') return false;
  const one = (name: string): Promise<boolean> => {
    let p: Promise<boolean>;
    try {
      p = Promise.resolve(perms.query!({ name })).then(
        (s) => s?.state === 'denied',
        () => false,
      );
    } catch {
      return Promise.resolve(false);
    }
    return withTimeout(p, 500, false);
  };
  const results = await Promise.all(names.map(one));
  return results.some((d) => d);
}

// ───────────────────────────── browser sensor source ─────────────────────────────

export interface BrowserSourceOptions {
  /** RelativeOrientationSensor → onQuaternion (default true) */
  quaternion?: boolean;
  /** AbsoluteOrientationSensor → onQuaternionAbs (default true; needs listener.onQuaternionAbs) */
  absoluteQuaternion?: boolean;
  /** 'deviceorientationabsolute' → onOrientationAbs (default true; needs listener.onOrientationAbs) */
  orientationAbs?: boolean;
  /** Magnetometer → onMagnetometer (default true; needs listener.onMagnetometer) */
  magnetometer?: boolean;
  /** screen orientation changes → onScreen (default true; needs listener.onScreen) */
  screenEvents?: boolean;
  /** requested rate of every Generic Sensor (default 60) */
  frequencyHz?: number;
}

/** Milliseconds without a data-carrying devicemotion event before onError('no_events'). */
export const NO_EVENTS_TIMEOUT_MS = 1500;

/** Permission names the AbsoluteOrientationSensor needs (Generic Sensor API). */
const ABSOLUTE_SENSOR_PERMISSIONS = ['accelerometer', 'gyroscope', 'magnetometer'] as const;

/** listener.onWarning, which SensorListener (types.ts) does not declare; see trialController.SensorWarningListener. */
function warnListener(listener: SensorListener, code: string): void {
  const fn = (listener as SensorListener & { onWarning?: unknown }).onWarning;
  if (typeof fn !== 'function') return;
  try {
    (fn as (code: string) => void).call(listener, code);
  } catch {
    // a warning must never break the source
  }
}

function readVec3(v: unknown): Vec3 | null {
  if (v === null || typeof v !== 'object') return null;
  const o = v as { x?: unknown; y?: unknown; z?: unknown };
  const x = finiteOrNull(o.x);
  const y = finiteOrNull(o.y);
  const z = finiteOrNull(o.z);
  return x === null || y === null || z === null ? null : [x, y, z];
}

function readEuler(v: unknown): Euler | null {
  if (v === null || typeof v !== 'object') return null;
  const o = v as { alpha?: unknown; beta?: unknown; gamma?: unknown };
  const alpha = finiteOrNull(o.alpha);
  const beta = finiteOrNull(o.beta);
  const gamma = finiteOrNull(o.gamma);
  return alpha === null || beta === null || gamma === null ? null : { alpha, beta, gamma };
}

export function createBrowserSensorSource(opts?: BrowserSourceOptions): SensorSource {
  const wantQuaternion = opts?.quaternion !== false;
  const wantAbsQuaternion = opts?.absoluteQuaternion !== false;
  const wantOrientationAbs = opts?.orientationAbs !== false;
  const wantMagnetometer = opts?.magnetometer !== false;
  const wantScreen = opts?.screenEvents !== false;
  const requestedHz = opts?.frequencyHz;
  const frequencyHz = typeof requestedHz === 'number' && Number.isFinite(requestedHz) && requestedHz > 0 ? requestedHz : 60;
  const clock: SourceClock = {
    now: perfNow,
    // Anchored on the current wall clock rather than performance.timeOrigin: the monotonic clock
    // drifts from (and may pause unlike) Date over a long-lived page.
    epochAt: (t: number) => Date.now() - (perfNow() - t),
  };
  let phase: 'idle' | 'running' | 'stopped' = 'idle';
  const cleanups: (() => void)[] = [];

  function stop(): void {
    if (phase === 'stopped') return;
    phase = 'stopped';
    for (const fn of cleanups.splice(0)) {
      try {
        fn();
      } catch {
        // keep removing the rest
      }
    }
  }

  function start(listener: SensorListener): Promise<void> {
    if (phase !== 'idle') throw new Error('BrowserSensorSource.start() called twice (sources are single-use)');
    phase = 'running';
    const g = G();

    const report = (e: SensorError): void => {
      if (phase === 'running') listener.onError?.(e);
    };
    // Errors detected during start() go out on a later task so the caller finishes starting first.
    const reportLater = (e: SensorError): void => {
      const timer = setTimeout(() => report(e), 0);
      cleanups.push(() => clearTimeout(timer));
    };

    const target = g.window;
    if (!target || typeof target.addEventListener !== 'function') {
      reportLater({ code: 'not_supported', message: 'No window: motion events are unavailable here' });
      return Promise.resolve();
    }
    if (!isSecure()) {
      reportLater({ code: 'insecure_context', message: 'Motion sensors need HTTPS (secure context)' });
    } else if (g.DeviceMotionEvent === undefined || g.DeviceMotionEvent === null) {
      reportLater({ code: 'not_supported', message: 'DeviceMotionEvent is not supported by this browser' });
    }

    let gotData = false;
    let emptyEvents = 0;
    let noEventsTimer: ReturnType<typeof setTimeout> | undefined;
    const armNoEvents = (): void => {
      noEventsTimer = setTimeout(() => {
        noEventsTimer = undefined;
        if (phase !== 'running' || gotData) return;
        if (isDocumentHidden()) {
          armNoEvents(); // hidden pages get no events: judge the sensor once we are visible again
          return;
        }
        report({
          code: 'no_events',
          message:
            `No devicemotion data within ${NO_EVENTS_TIMEOUT_MS} ms` +
            (emptyEvents > 0 ? ` (${emptyEvents} events without values)` : ''),
        });
      }, NO_EVENTS_TIMEOUT_MS);
    };
    armNoEvents();
    cleanups.push(() => clearTimeout(noEventsTimer));

    const onMotion = (ev: Event): void => {
      const t = perfNow(); // first statement: this IS the sample time
      if (phase !== 'running') return;
      const e = ev as Partial<DeviceMotionEvent>;
      const sample: MotionSample = {
        t,
        ts: finiteOrNull(ev.timeStamp),
        acc: readVec3(e.acceleration),
        accg: readVec3(e.accelerationIncludingGravity),
        rr: readEuler(e.rotationRate),
        interval: finiteOrNull(e.interval),
      };
      if (sample.acc || sample.accg || sample.rr) gotData = true;
      else emptyEvents++;
      listener.onMotion(sample);
    };

    const onOrientation = (ev: Event): void => {
      const t = perfNow();
      if (phase !== 'running') return;
      listener.onOrientation(orientationSample(ev, t));
    };

    target.addEventListener('devicemotion', onMotion);
    target.addEventListener('deviceorientation', onOrientation);
    cleanups.push(() => {
      target.removeEventListener('devicemotion', onMotion);
      target.removeEventListener('deviceorientation', onOrientation);
    });

    // ── schema 1.2: absolute device orientation events (Chrome) ──
    if (wantOrientationAbs && typeof listener.onOrientationAbs === 'function' && 'ondeviceorientationabsolute' in target) {
      const onOrientationAbs = (ev: Event): void => {
        const t = perfNow();
        if (phase !== 'running') return;
        listener.onOrientationAbs?.(orientationSample(ev, t));
      };
      target.addEventListener('deviceorientationabsolute', onOrientationAbs);
      cleanups.push(() => target.removeEventListener('deviceorientationabsolute', onOrientationAbs));
    }

    // ── schema 1.2: screen orientation changes ──
    if (wantScreen && typeof listener.onScreen === 'function') attachScreenEvents(target, listener);

    // ── Generic Sensors (each optional, set up in parallel; start() resolves when all are decided) ──
    const pending: Promise<void>[] = [];
    const nav = g.navigator;
    const RelCtor = g.RelativeOrientationSensor;
    if (wantQuaternion && typeof RelCtor === 'function') {
      pending.push(
        genericSensorsGranted(nav).then((granted) => {
          if (!granted || phase !== 'running') return;
          startGenericSensor(
            RelCtor as GenericSensorCtor,
            (sensor, t) => {
              const q = readQuaternion(sensor);
              if (q) listener.onQuaternion?.({ t, q, sensor: 'RelativeOrientationSensor' });
            },
            // NotReadableError (no gyroscope), NotAllowedError...: the quaternion is optional, so an
            // error only ends this stream; devicemotion keeps going and no onError is raised.
            () => warnListener(listener, 'quaternion_unavailable'),
          );
        }),
      );
    }
    const AbsCtor = g.AbsoluteOrientationSensor;
    if (wantAbsQuaternion && typeof AbsCtor === 'function' && typeof listener.onQuaternionAbs === 'function') {
      pending.push(
        anyPermissionDenied(nav, ABSOLUTE_SENSOR_PERMISSIONS).then((denied) => {
          if (phase !== 'running') return;
          if (denied) {
            warnListener(listener, 'abs_orientation_denied');
            return;
          }
          startGenericSensor(
            AbsCtor as GenericSensorCtor,
            (sensor, t) => {
              const q = readQuaternion(sensor);
              if (q) listener.onQuaternionAbs?.({ t, q, sensor: 'AbsoluteOrientationSensor' });
            },
            () => warnListener(listener, 'abs_orientation_unavailable'),
          );
        }),
      );
    }
    const MagCtor = g.Magnetometer;
    if (wantMagnetometer && typeof MagCtor === 'function' && typeof listener.onMagnetometer === 'function') {
      pending.push(
        anyPermissionDenied(nav, ['magnetometer']).then((denied) => {
          if (phase !== 'running') return;
          if (denied) {
            warnListener(listener, 'magnetometer_denied');
            return;
          }
          startGenericSensor(
            MagCtor as GenericSensorCtor,
            (sensor, t) => {
              const x = finiteOrNull(sensor.x);
              const y = finiteOrNull(sensor.y);
              const z = finiteOrNull(sensor.z);
              if (x === null && y === null && z === null) return; // no reading yet
              listener.onMagnetometer?.({ t, x, y, z });
            },
            () => warnListener(listener, 'magnetometer_unavailable'),
          );
        }),
      );
    }
    if (pending.length === 0) return Promise.resolve();
    return Promise.all(pending.map((p) => p.catch(() => undefined))).then(() => undefined);
  }

  /** screen.orientation 'change' (else the legacy window 'orientationchange') → onScreen. */
  function attachScreenEvents(target: EventTarget, listener: SensorListener): void {
    const onChange = (): void => {
      const t = perfNow();
      if (phase !== 'running') return;
      const s = getScreenOrientation();
      listener.onScreen?.({ t, type: s.type, angle: s.angle });
    };
    try {
      const so = G().screen?.orientation;
      if (so && typeof so.addEventListener === 'function' && typeof so.removeEventListener === 'function') {
        so.addEventListener('change', onChange);
        cleanups.push(() => so.removeEventListener?.('change', onChange));
        return;
      }
    } catch {
      // fall back to the window event below
    }
    if ('onorientationchange' in target) {
      target.addEventListener('orientationchange', onChange);
      cleanups.push(() => target.removeEventListener('orientationchange', onChange));
    }
  }

  /**
   * Starts one Generic Sensor at frequencyHz in the device frame. `onFail` runs once if it cannot
   * run: the constructor or start() throws (SecurityError from a permissions policy, ...) or the
   * sensor fires 'error' (NotReadableError: no such hardware, NotAllowedError, ...). Readings are
   * delivered with t = performance.now() taken first thing in the handler; stop() stops the sensor.
   */
  function startGenericSensor(
    SensorCtor: GenericSensorCtor,
    onReading: (sensor: GenericSensorLike, t: number) => void,
    onFail: () => void,
  ): void {
    let sensor: GenericSensorLike;
    try {
      sensor = new SensorCtor({ frequency: frequencyHz, referenceFrame: 'device' });
    } catch {
      onFail();
      return;
    }
    let active = true;
    const reading = (): void => {
      const t = perfNow();
      if (!active || phase !== 'running') return;
      onReading(sensor, t);
    };
    const stopSensor = (): void => {
      if (!active) return;
      active = false;
      sensor.removeEventListener('reading', reading);
      sensor.removeEventListener('error', failed);
      try {
        sensor.stop();
      } catch {
        // already stopped
      }
    };
    function failed(): void {
      if (!active) return;
      stopSensor();
      onFail();
    }
    sensor.addEventListener('reading', reading);
    sensor.addEventListener('error', failed);
    cleanups.push(stopSensor);
    try {
      sensor.start();
    } catch {
      failed();
    }
  }

  return { kind: 'browser', clock, start, stop };
}

/** A deviceorientation(absolute) event as an OrientationSample (iOS: webkitCompassHeading / Accuracy). */
function orientationSample(ev: Event, t: number): OrientationSample {
  const e = ev as Partial<DeviceOrientationEvent> & { webkitCompassHeading?: unknown; webkitCompassAccuracy?: unknown };
  return {
    t,
    ts: finiteOrNull(ev.timeStamp),
    alpha: finiteOrNull(e.alpha),
    beta: finiteOrNull(e.beta),
    gamma: finiteOrNull(e.gamma),
    absolute: typeof e.absolute === 'boolean' ? e.absolute : null,
    compass: finiteOrNull(e.webkitCompassHeading),
    compass_accuracy: finiteOrNull(e.webkitCompassAccuracy),
  };
}

/** sensor.quaternion as [x, y, z, w], or null when absent or not finite. */
function readQuaternion(sensor: GenericSensorLike): [number, number, number, number] | null {
  const q = sensor.quaternion;
  if (!q || q.length !== 4) return null;
  const x = q[0];
  const y = q[1];
  const z = q[2];
  const w = q[3];
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) || !Number.isFinite(w)) return null;
  return [x, y, z, w];
}

// ───────────────────────────── touches ─────────────────────────────

export interface TouchReading {
  x: number | null;
  y: number | null;
  details: TouchDetails;
}

/**
 * clientX / clientY of a DOM Touch (or anything shaped like one) and its schema 1.2 contact details:
 * radiusX / radiusY (CSS px), force (0..1), rotationAngle (deg) and the position normalised by the
 * viewport (`viewport`, default window.innerWidth / innerHeight; clamped to [0, 1], null without a
 * positive size). Missing, non-numeric or non-finite fields become null; never throws. Typical use:
 *   const r = readTouch(e.changedTouches[0] ?? e.touches[0]);
 *   controller.addTouch(type, e.touches.length, r.x, r.y, undefined, r.details);
 */
export function readTouch(touch: unknown, viewport?: { width: number; height: number } | null): TouchReading {
  const none: TouchDetails = { radius_x: null, radius_y: null, force: null, rotation_angle: null, x_norm: null, y_norm: null };
  try {
    if (touch === null || typeof touch !== 'object') return { x: null, y: null, details: none };
    const o = touch as { clientX?: unknown; clientY?: unknown; radiusX?: unknown; radiusY?: unknown; force?: unknown; rotationAngle?: unknown };
    const x = finiteOrNull(o.clientX);
    const y = finiteOrNull(o.clientY);
    const g = G();
    const w = finiteOrNull(viewport ? viewport.width : g.innerWidth);
    const h = finiteOrNull(viewport ? viewport.height : g.innerHeight);
    const norm = (v: number | null, size: number | null): number | null =>
      v === null || size === null || !(size > 0) ? null : Math.min(1, Math.max(0, v / size));
    return {
      x,
      y,
      details: {
        radius_x: finiteOrNull(o.radiusX),
        radius_y: finiteOrNull(o.radiusY),
        force: finiteOrNull(o.force),
        rotation_angle: finiteOrNull(o.rotationAngle),
        x_norm: norm(x, w),
        y_norm: norm(y, h),
      },
    };
  } catch {
    return { x: null, y: null, details: none };
  }
}

// ───────────────────────────── rate measurement ─────────────────────────────

export interface RateSummary {
  n: number;
  /** (n − 1) / (t_last − t_first) · 1000; null for fewer than 2 samples or a zero span */
  hz: number | null;
  dt_median: number | null;
  dt_p95: number | null;
}

/** Linear-interpolation quantile of an ascending array (numpy's default method). */
function quantileSorted(sorted: number[], q: number): number | null {
  if (sorted.length === 0) return null;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Sampling-rate summary of a list of sample times (ms) in arrival order. */
export function rateSummary(times: number[]): RateSummary {
  const n = times.length;
  if (n < 2) return { n, hz: null, dt_median: null, dt_p95: null };
  const dts: number[] = [];
  for (let i = 1; i < n; i++) dts.push(times[i] - times[i - 1]);
  dts.sort((a, b) => a - b);
  const span = times[n - 1] - times[0];
  return {
    n,
    hz: span > 0 ? ((n - 1) / span) * 1000 : null,
    dt_median: quantileSorted(dts, 0.5),
    dt_p95: quantileSorted(dts, 0.95),
  };
}

function round3(v: number | null): number | null {
  return v === null ? null : Math.round(v * 1000) / 1000;
}

export function measureRates(source: SensorSource, durationMs: number): Promise<MeasuredRates> {
  const motionT: number[] = [];
  const orientationT: number[] = [];
  const quaternionT: number[] = [];
  let acc = false;
  let accg = false;
  let rotation = false;
  let intervalRaw: number | null = null;
  const t0 = source.clock.now();
  const limit = Number.isFinite(durationMs) && durationMs > 0 ? durationMs : 0;

  return new Promise<MeasuredRates>((resolve, reject) => {
    let done = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (): void => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      source.stop();
      const elapsed = source.clock.now() - t0;
      const motion = rateSummary(motionT);
      resolve({
        duration_ms: round3(Math.max(0, elapsed)) ?? 0,
        motion_hz: round3(motion.hz),
        orientation_hz: round3(rateSummary(orientationT).hz),
        quaternion_hz: round3(rateSummary(quaternionT).hz),
        motion_dt_ms_median: round3(motion.dt_median),
        motion_dt_ms_p95: round3(motion.dt_p95),
        acc_available: acc,
        accg_available: accg,
        rotation_available: rotation,
        interval_raw: intervalRaw,
      });
    };
    const fail = (e: unknown): void => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      source.stop();
      reject(e);
    };
    const listener: SensorListener = {
      onMotion: (s) => {
        if (done) return;
        motionT.push(s.t);
        if (s.acc) acc = true;
        if (s.accg) accg = true;
        if (s.rr) rotation = true;
        if (intervalRaw === null && s.interval !== null) intervalRaw = s.interval;
        // Source-time check: an offline simulated source delivers everything inside start().
        if (s.t - t0 >= limit) finish();
      },
      onOrientation: (s) => {
        if (!done) orientationT.push(s.t);
      },
      onQuaternion: (s) => {
        if (!done) quaternionT.push(s.t);
      },
      onError: () => {
        // 'no_events' etc.: keep waiting, the summary (motion_hz null) tells the story.
      },
    };
    timer = setTimeout(finish, limit);
    try {
      source.start(listener).catch(fail);
    } catch (e) {
      fail(e);
    }
  });
}

// ───────────────────────────── device info ─────────────────────────────

export interface UaEnv {
  /** navigator.maxTouchPoints (iPadOS "desktop" mode claims to be a Mac but has touch) */
  maxTouchPoints?: number;
  /** launched from the home screen (iOS Home Screen web apps have no "Safari/" token) */
  standalone?: boolean;
  /** navigator.brave exists (Brave has a plain Chrome UA) */
  brave?: boolean;
}

export interface ParsedUserAgent {
  platform: Platform;
  browser: string;
  os_version: string | null;
  model_hint: string | null;
  in_app_browser: string | null;
  /** Chrome's reduced Android UA ("Android 10; K"): the version and model in it are placeholders */
  reduced_ua: boolean;
}

/** [detector, id, display name, version pattern] — checked in order, before the browser table. */
const IN_APP_BROWSERS: [RegExp, string, string, RegExp | null][] = [
  [/\bInstagram\b/, 'instagram', 'Instagram', /\bInstagram ([\d.]+)/],
  [/FBAN\/|FBAV\/|FB_IAB\/|FBIOS|FB4A/, 'facebook', 'Facebook', /FBAV\/([\d.]+)/],
  [/musical_ly|BytedanceWebview|TikTok|\btrill_/, 'tiktok', 'TikTok', /musical_ly_([\d.]+)/],
  [/\bSnapchat\b/, 'snapchat', 'Snapchat', /Snapchat\/([\d.]+)/],
  [/\bLine\/\d/, 'line', 'LINE', /\bLine\/([\d.]+)/],
  [/\bGSA\/\d/, 'google_app', 'Google app', /\bGSA\/([\d.]+)/],
  [/LinkedInApp/, 'linkedin', 'LinkedIn', /LinkedInApp\/([\d.]+)/],
  [/MicroMessenger/, 'wechat', 'WeChat', /MicroMessenger\/([\d.]+)/],
  [/\bPinterest\b/, 'pinterest', 'Pinterest', /Pinterest\/([\d.]+)/],
  // WhatsApp opens business links (and, in an iOS beta of Oct 2026, any link) in its own browser
  [/\bWhatsApp\b/, 'whatsapp', 'WhatsApp', /WhatsApp\/([\d.]+)/],
];

/** Order matters: wrapper browsers carry the tokens of the engine browser after their own. */
const BROWSERS: [RegExp, string][] = [
  [/EdgiOS\/([\d.]+)/, 'Edge iOS'],
  [/CriOS\/([\d.]+)/, 'Chrome iOS'],
  [/FxiOS\/([\d.]+)/, 'Firefox iOS'],
  [/EdgA\/([\d.]+)/, 'Edge Android'],
  [/\bEdg\/([\d.]+)/, 'Edge'],
  [/\bOPR\/([\d.]+)/, 'Opera'],
  [/SamsungBrowser\/([\d.]+)/, 'Samsung Internet'],
  [/MiuiBrowser\/([\d.]+)/, 'MIUI Browser'],
  [/HuaweiBrowser\/([\d.]+)/, 'Huawei Browser'],
  [/YaBrowser\/([\d.]+)/, 'Yandex Browser'],
  [/UCBrowser\/([\d.]+)/, 'UC Browser'],
  [/\bFirefox\/([\d.]+)/, 'Firefox'],
  [/\b(?:Chrome|Chromium)\/([\d.]+)/, 'Chrome'],
  [/\bVersion\/([\d.]+).*\bSafari\//, 'Safari'],
];

function versionOf(re: RegExp | null, ua: string): string | null {
  if (!re) return null;
  const m = re.exec(ua);
  return m && m[1] ? m[1] : null;
}

function major(v: string | null): number | null {
  if (!v) return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

/** Device model from the "(Linux; Android 14; SM-S918B Build/…; wv)" section; null if absent/reduced. */
function androidModelFromUa(ua: string): string | null {
  const section = /\(([^)]*Android[^)]*)\)/.exec(ua);
  if (!section) return null;
  const tokens = section[1].split(';').map((s) => s.trim());
  const at = tokens.findIndex((s) => /^Android\b/.test(s));
  if (at < 0) return null;
  for (const raw of tokens.slice(at + 1)) {
    // locale ("en-us"), form factor, webview flag, legacy "U", Firefox's rv: are not models
    if (!raw || /^(?:[a-z]{2}[-_][a-z]{2}|Mobile|Tablet|TV|wv|U|rv:.*)$/i.test(raw)) continue;
    const model = raw
      .replace(/\s*Build\/.*$/, '')
      .replace(/^SAMSUNG\s+/i, '')
      .trim();
    return model && model !== 'K' ? model : null;
  }
  return null;
}

/** Pure UA parsing (exported for tests). Engine detection needs the live prototype, see getDeviceInfo. */
export function parseUserAgent(ua: string, env: UaEnv = {}): ParsedUserAgent {
  const touchMac = /\bMacintosh\b/.test(ua) && (env.maxTouchPoints ?? 0) > 1; // iPadOS desktop mode
  const iosDevice = /\b(iPhone|iPad|iPod)\b/.exec(ua);
  const platform: Platform = iosDevice || touchMac ? 'ios' : /\bAndroid\b/.test(ua) ? 'android' : 'other';
  const reduced_ua = platform === 'android' && /\bAndroid 10; K\)/.test(ua);
  const safariVersion = versionOf(/\bVersion\/([\d.]+)/, ua);

  // ── in-app browser / webview ──
  let in_app_browser: string | null = null;
  let browser: string | null = null;
  for (const [detect, id, name, ver] of IN_APP_BROWSERS) {
    if (!detect.test(ua)) continue;
    in_app_browser = id;
    const v = versionOf(ver, ua);
    browser = v ? `${name} ${v}` : name;
    break;
  }
  if (!in_app_browser && platform === 'android' && /;\s*wv\)/.test(ua)) {
    in_app_browser = 'android_webview';
    const v = versionOf(/\bChrome\/([\d.]+)/, ua);
    browser = v ? `Android WebView ${v}` : 'Android WebView';
  }

  // ── regular browsers ──
  if (!browser) {
    for (const [re, name] of BROWSERS) {
      const m = re.exec(ua);
      if (!m) continue;
      browser = `${env.brave && (name === 'Chrome' || name === 'Safari') ? 'Brave' : name} ${m[1]}`;
      break;
    }
  }
  if (!browser && platform === 'ios') {
    if (env.standalone) {
      browser = 'Safari'; // Home Screen web apps drop the Version/… Safari/… tokens
    } else if (!/\bSafari\//.test(ua)) {
      in_app_browser = 'ios_webview'; // a bare WKWebView inside some app
      browser = 'iOS WebView';
    }
  }

  // ── OS version ──
  let os_version: string | null = null;
  if (platform === 'ios') {
    if (touchMac) {
      // The Mac UA freezes "10_15_7"; iPadOS Safari's version is the iPadOS version.
      os_version = safariVersion;
    } else {
      const m = /\bOS (\d+(?:_\d+)*) like Mac OS X/.exec(ua);
      os_version = m ? m[1].replace(/_/g, '.') : null;
      // Safari 26+ freezes the OS in its UA (18_6); its own version equals the iOS version.
      const sv = major(safariVersion);
      const ov = major(os_version);
      if (sv !== null && sv >= 26 && (ov === null || sv > ov)) os_version = safariVersion;
    }
  } else if (platform === 'android') {
    os_version = reduced_ua ? null : versionOf(/\bAndroid (\d+(?:\.\d+)*)/, ua);
  } else {
    const win = versionOf(/\bWindows NT (\d+\.\d+)/, ua);
    const mac = versionOf(/\bMac OS X (\d+(?:[_.]\d+)*)/, ua);
    os_version = win ?? (mac ? mac.replace(/_/g, '.') : null);
  }

  // ── model hint ──
  let model_hint: string | null = null;
  if (platform === 'ios') {
    // Instagram/Facebook webviews append the hardware model ("iPhone15,2"); Safari never does.
    model_hint =
      versionOf(/\((i(?:Phone|Pad|Pod)\d+,\d+);/, ua) ??
      versionOf(/FBDV\/([^;\]]+)/, ua) ??
      (touchMac ? 'iPad' : iosDevice ? iosDevice[1] : null);
  } else if (platform === 'android') {
    model_hint = reduced_ua ? null : androidModelFromUa(ua);
  }

  return {
    platform,
    browser: browser ?? 'unknown',
    os_version,
    model_hint,
    in_app_browser,
    reduced_ua,
  };
}

function detectEngine(ua: string, hasUaData: boolean): NonNullable<DeviceInfo['engine']> {
  if (hasWebkitCompass()) return 'webkit'; // every iOS browser; the only reliable signal
  if (/\bGecko\/\d/.test(ua) && /\bFirefox\//.test(ua)) return 'gecko';
  if (hasUaData || /\b(?:Chrome|Chromium)\/\d/.test(ua)) return 'blink';
  return 'unknown';
}

function isStandalone(): boolean {
  const g = G();
  try {
    if (typeof g.matchMedia === 'function') {
      if (g.matchMedia('(display-mode: standalone)').matches || g.matchMedia('(display-mode: fullscreen)').matches) {
        return true;
      }
    }
  } catch {
    // very old engines throw on unknown media features
  }
  return g.navigator?.standalone === true;
}

async function highEntropyHints(nav: LooseNavigator | undefined): Promise<{ model: string | null; platformVersion: string | null }> {
  const none = { model: null, platformVersion: null };
  const uad = nav?.userAgentData;
  if (!uad || typeof uad.getHighEntropyValues !== 'function') return none;
  try {
    const values = await withTimeout(uad.getHighEntropyValues(['model', 'platformVersion']), 500, null);
    if (!values) return none;
    const text = (v: unknown): string | null => (typeof v === 'string' && v.trim() !== '' ? v.trim() : null);
    return { model: text(values.model), platformVersion: text(values.platformVersion) };
  } catch {
    return none;
  }
}

export async function getDeviceInfo(): Promise<DeviceInfo> {
  const g = G();
  const nav = g.navigator;
  const ua = typeof nav?.userAgent === 'string' ? nav.userAgent : '';
  const standalone = isStandalone();
  const parsed = parseUserAgent(ua, {
    maxTouchPoints: finiteOrNull(nav?.maxTouchPoints) ?? 0,
    standalone,
    brave: nav?.brave !== undefined && nav?.brave !== null,
  });
  const hints = await highEntropyHints(nav);
  const screenW = finiteOrNull(g.screen?.width);
  const screenH = finiteOrNull(g.screen?.height);
  return {
    user_agent: ua,
    platform: parsed.platform,
    browser: parsed.browser,
    // UA-CH is the only source of the real Android version/model behind the reduced UA.
    os_version: hints.platformVersion ?? parsed.os_version,
    model_hint: hints.model ?? parsed.model_hint,
    standalone,
    screen_w: screenW ?? 0,
    screen_h: screenH ?? 0,
    dpr: finiteOrNull(g.devicePixelRatio) ?? 1,
    language: typeof nav?.language === 'string' ? nav.language : '',
    hardware_concurrency: finiteOrNull(nav?.hardwareConcurrency),
    device_memory_gb: finiteOrNull(nav?.deviceMemory),
    time_origin_ms: timeOrigin(),
    engine: detectEngine(ua, !!nav?.userAgentData),
    in_app_browser: parsed.in_app_browser,
  };
}

// ───────────────────────────── screen orientation ─────────────────────────────

const LEGACY_ORIENTATION_TYPES: Record<number, string> = {
  0: 'portrait-primary',
  90: 'landscape-primary',
  180: 'portrait-secondary',
  270: 'landscape-secondary',
};

export function getScreenOrientation(): { type: string | null; angle: number | null } {
  try {
    const g = G();
    const so = g.screen?.orientation;
    if (so && typeof so.type === 'string') return { type: so.type, angle: finiteOrNull(so.angle) };
    const legacy = finiteOrNull(g.orientation);
    if (legacy !== null) {
      // window.orientation: 90 = rotated counter-clockwise, −90 = clockwise → screen.orientation angles.
      const angle = ((legacy % 360) + 360) % 360;
      return { type: LEGACY_ORIENTATION_TYPES[angle] ?? null, angle };
    }
  } catch {
    // fall through
  }
  return { type: null, angle: null };
}
