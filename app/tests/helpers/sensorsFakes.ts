/**
 * Fake browser globals for the sensors/cues tests (they run in Node). Install them with
 * vi.stubGlobal and undo with vi.unstubAllGlobals() in afterEach.
 */
import { vi } from 'vitest';

type Callback = EventListenerOrEventListenerObject | null;

/** EventTarget that can tell how many listeners are attached (Node's cannot). */
export class TrackingTarget extends EventTarget {
  private readonly listeners = new Map<string, Set<unknown>>();

  override addEventListener(type: string, cb: Callback, options?: AddEventListenerOptions | boolean): void {
    if (cb) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type)!.add(cb);
    }
    super.addEventListener(type, cb, options);
  }

  override removeEventListener(type: string, cb: Callback, options?: EventListenerOptions | boolean): void {
    if (cb) this.listeners.get(type)?.delete(cb);
    super.removeEventListener(type, cb, options);
  }

  listenerCount(type?: string): number {
    if (type !== undefined) return this.listeners.get(type)?.size ?? 0;
    let n = 0;
    for (const set of this.listeners.values()) n += set.size;
    return n;
  }
}

/** An Event whose extra fields (and timeStamp) are plain own properties. */
export function fakeEvent(type: string, fields: Record<string, unknown> = {}): Event {
  const ev = new Event(type);
  for (const [k, v] of Object.entries(fields)) Object.defineProperty(ev, k, { value: v, configurable: true });
  return ev;
}

export interface MotionFields {
  acceleration?: unknown;
  accelerationIncludingGravity?: unknown;
  rotationRate?: unknown;
  interval?: unknown;
  timeStamp?: number;
}

export function motionEvent(f: MotionFields): Event {
  return fakeEvent('devicemotion', f as Record<string, unknown>);
}

export function orientationEvent(f: Record<string, unknown>): Event {
  return fakeEvent('deviceorientation', f);
}

export type PermissionBehavior = 'granted' | 'denied' | 'reject_not_allowed' | 'throw' | 'never';

/** Implements a static requestPermission() that insists on being called as a method. */
function permissionFn(owner: object, behavior: PermissionBehavior) {
  return vi.fn(function (this: unknown): Promise<string> {
    if (this !== owner) throw new TypeError('Illegal invocation');
    switch (behavior) {
      case 'granted':
      case 'denied':
        return Promise.resolve(behavior);
      case 'reject_not_allowed':
        return Promise.reject(new DOMException('Requesting device orientation access requires a user gesture', 'NotAllowedError'));
      case 'throw':
        throw new TypeError('boom');
      case 'never':
        return new Promise<string>(() => undefined);
    }
  });
}

export interface MotionInterfaces {
  DeviceMotionEvent: { new (type: string): Event; requestPermission?: ReturnType<typeof permissionFn> };
  DeviceOrientationEvent: { new (type: string): Event; requestPermission?: ReturnType<typeof permissionFn> };
}

/**
 * Fake DeviceMotionEvent / DeviceOrientationEvent interface objects.
 * webkit: DeviceOrientationEvent.prototype gets webkitCompassHeading (every iOS browser has it).
 */
export function fakeMotionInterfaces(opts: {
  motionPermission?: PermissionBehavior | null;
  orientationPermission?: PermissionBehavior | null;
  webkit?: boolean;
} = {}): MotionInterfaces {
  class FakeDeviceMotionEvent extends Event {}
  class FakeDeviceOrientationEvent extends Event {}
  const dme = FakeDeviceMotionEvent as MotionInterfaces['DeviceMotionEvent'];
  const doe = FakeDeviceOrientationEvent as MotionInterfaces['DeviceOrientationEvent'];
  if (opts.motionPermission) dme.requestPermission = permissionFn(dme, opts.motionPermission);
  if (opts.orientationPermission) doe.requestPermission = permissionFn(doe, opts.orientationPermission);
  if (opts.webkit) {
    Object.defineProperty(FakeDeviceOrientationEvent.prototype, 'webkitCompassHeading', {
      get: () => null,
      configurable: true,
    });
  }
  return { DeviceMotionEvent: dme, DeviceOrientationEvent: doe };
}

/** RelativeOrientationSensor fake: records constructions, emits readings/errors on demand. */
export function fakeOrientationSensorClass(opts: { throwOnConstruct?: boolean } = {}) {
  const instances: FakeSensor[] = [];
  class FakeSensor extends TrackingTarget {
    quaternion: number[] | null = null;
    started = 0;
    stopped = 0;
    constructor(readonly options: { frequency?: number; referenceFrame?: string }) {
      super();
      if (opts.throwOnConstruct) throw new DOMException('blocked by permissions policy', 'SecurityError');
      instances.push(this);
    }
    start(): void {
      this.started++;
    }
    stop(): void {
      this.stopped++;
    }
    emitReading(q: number[] | null): void {
      this.quaternion = q;
      this.dispatchEvent(new Event('reading'));
    }
    emitError(): void {
      this.dispatchEvent(new Event('error'));
    }
  }
  return { Sensor: FakeSensor, instances };
}

/** Magnetometer fake (Generic Sensor API, µT readings in x / y / z); same controls as above. */
export function fakeMagnetometerClass(opts: { throwOnConstruct?: boolean; throwOnStart?: boolean } = {}) {
  const instances: FakeMagnetometer[] = [];
  class FakeMagnetometer extends TrackingTarget {
    x: number | null = null;
    y: number | null = null;
    z: number | null = null;
    started = 0;
    stopped = 0;
    constructor(readonly options: { frequency?: number; referenceFrame?: string }) {
      super();
      if (opts.throwOnConstruct) throw new DOMException('blocked by permissions policy', 'SecurityError');
      instances.push(this);
    }
    start(): void {
      this.started++;
      if (opts.throwOnStart) throw new DOMException('not allowed', 'NotAllowedError');
    }
    stop(): void {
      this.stopped++;
    }
    emitReading(x: number | null, y: number | null, z: number | null): void {
      this.x = x;
      this.y = y;
      this.z = z;
      this.dispatchEvent(new Event('reading'));
    }
    emitError(): void {
      this.dispatchEvent(new Event('error'));
    }
  }
  return { Sensor: FakeMagnetometer, instances };
}

/** screen.orientation fake: rotate() changes type / angle and fires 'change'. */
export class FakeScreenOrientation extends TrackingTarget {
  type = 'portrait-primary';
  angle = 0;
  rotate(type: string, angle: number): void {
    this.type = type;
    this.angle = angle;
    this.dispatchEvent(new Event('change'));
  }
}

export function fakePermissionsApi(states: Record<string, string | 'throw'>) {
  return {
    query: vi.fn((d: { name: string }) => {
      const s = states[d.name];
      if (s === undefined || s === 'throw') return Promise.reject(new TypeError(`'${d.name}' is not a valid PermissionName`));
      return Promise.resolve({ state: s });
    }),
  };
}

/** Minimal document with a settable visibility state that fires 'visibilitychange'. */
export class FakeDocument extends TrackingTarget {
  visibilityState: 'visible' | 'hidden' = 'visible';
  get hidden(): boolean {
    return this.visibilityState === 'hidden';
  }
  setVisibility(v: 'visible' | 'hidden'): void {
    this.visibilityState = v;
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

// ───────────────────────────── Web Audio ─────────────────────────────

export class FakeParam {
  readonly events: [kind: 'set' | 'ramp', value: number, time: number][] = [];
  setValueAtTime(value: number, time: number): void {
    this.events.push(['set', value, time]);
  }
  linearRampToValueAtTime(value: number, time: number): void {
    this.events.push(['ramp', value, time]);
  }
}

export class FakeOscillator {
  type = 'sine';
  readonly frequency = new FakeParam();
  onended: (() => void) | null = null;
  startAt: number | null = null;
  stopAt: number | null = null;
  readonly connections: unknown[] = [];
  connect(dest: unknown): unknown {
    this.connections.push(dest);
    return dest;
  }
  disconnect(): void {}
  start(when?: number): void {
    this.startAt = when ?? 0;
  }
  stop(when?: number): void {
    this.stopAt = when ?? 0;
  }
}

export class FakeGainNode {
  readonly gain = new FakeParam();
  readonly connections: unknown[] = [];
  connect(dest: unknown): unknown {
    this.connections.push(dest);
    return dest;
  }
  disconnect(): void {}
}

/** A fresh AudioContext class per test (static state would leak between tests otherwise). */
/**
 * stuck: resume() never changes the state (iOS contexts stuck on 'interrupted', WebKit bug 273511);
 * suspend() then sets 'suspended' (unless stuck) and close() 'closed'.
 */
export function fakeAudioContextClass(
  opts: { throwOnConstruct?: boolean; initialState?: string; throwOnCreate?: boolean; stuck?: boolean } = {},
) {
  const instances: FakeAudioContext[] = [];
  class FakeAudioContext {
    state: string = opts.initialState ?? 'suspended';
    currentTime = 12.5;
    readonly destination = { kind: 'destination' };
    readonly oscillators: FakeOscillator[] = [];
    readonly gains: FakeGainNode[] = [];
    resumeCalls = 0;
    suspendCalls = 0;
    closeCalls = 0;
    /** the order of suspend / resume calls */
    readonly calls: string[] = [];
    silentStarts = 0;
    constructor() {
      if (opts.throwOnConstruct) throw new Error('AudioContext unavailable');
      instances.push(this);
    }
    createOscillator(): FakeOscillator {
      if (opts.throwOnCreate) throw new Error('createOscillator failed');
      const o = new FakeOscillator();
      this.oscillators.push(o);
      return o;
    }
    createGain(): FakeGainNode {
      const g = new FakeGainNode();
      this.gains.push(g);
      return g;
    }
    createBuffer(): object {
      return {};
    }
    createBufferSource() {
      const ctx = this;
      return {
        buffer: null as unknown,
        connect: (d: unknown) => d,
        disconnect: () => undefined,
        start: () => {
          ctx.silentStarts++;
        },
      };
    }
    resume(): Promise<void> {
      this.resumeCalls++;
      this.calls.push('resume');
      if (this.state !== 'closed' && !opts.stuck) this.state = 'running';
      return Promise.resolve();
    }
    suspend(): Promise<void> {
      this.suspendCalls++;
      this.calls.push('suspend');
      if (this.state !== 'closed' && !opts.stuck) this.state = 'suspended';
      return Promise.resolve();
    }
    close(): Promise<void> {
      this.closeCalls++;
      this.state = 'closed';
      return Promise.resolve();
    }
  }
  return { AudioContext: FakeAudioContext, instances };
}

// ───────────────────────────── speech ─────────────────────────────

export interface FakeVoice {
  name: string;
  lang: string;
  default?: boolean;
  localService?: boolean;
}

export interface FakeUtterance {
  text: string;
  lang: string;
  voice: FakeVoice | null;
  rate: number;
  pitch: number;
  volume: number;
}

/** speechSynthesis fake. asyncVoices: getVoices() is empty until loadVoices() fires 'voiceschanged'. */
export function fakeSpeech(voices: FakeVoice[], opts: { asyncVoices?: boolean } = {}) {
  let loaded = !opts.asyncVoices;
  const spoken: FakeUtterance[] = [];
  const target = new EventTarget();
  const synth = {
    speaking: false,
    pending: false,
    getVoices: vi.fn(() => (loaded ? voices.slice() : [])),
    speak: vi.fn((u: FakeUtterance) => {
      spoken.push(u);
    }),
    cancel: vi.fn(() => undefined),
    addEventListener: (type: string, fn: EventListener) => target.addEventListener(type, fn),
    removeEventListener: (type: string, fn: EventListener) => target.removeEventListener(type, fn),
  };
  class Utterance implements FakeUtterance {
    lang = '';
    voice: FakeVoice | null = null;
    rate = 1;
    pitch = 1;
    volume = 1;
    constructor(readonly text: string) {}
  }
  return {
    synth,
    Utterance,
    spoken,
    loadVoices(): void {
      loaded = true;
      target.dispatchEvent(new Event('voiceschanged'));
    },
  };
}

// ───────────────────────────── wake lock ─────────────────────────────

export class FakeSentinel extends EventTarget {
  released = false;
  readonly type = 'screen';
  readonly release = vi.fn(async (): Promise<void> => {
    this.markReleased();
  });
  /** What the browser does when the page is hidden. */
  systemRelease(): void {
    this.markReleased();
  }
  private markReleased(): void {
    if (this.released) return;
    this.released = true;
    this.dispatchEvent(new Event('release'));
  }
}

export function fakeWakeLock(opts: { reject?: boolean } = {}) {
  const sentinels: FakeSentinel[] = [];
  let reject = !!opts.reject;
  let gate: Promise<void> | null = null;
  const wakeLock = {
    request: vi.fn(async (type: string): Promise<FakeSentinel> => {
      if (type !== 'screen') throw new TypeError('bad type');
      if (gate) await gate;
      if (reject) throw new DOMException('The request is not allowed', 'NotAllowedError');
      const s = new FakeSentinel();
      sentinels.push(s);
      return s;
    }),
  };
  return {
    wakeLock,
    sentinels,
    setReject(v: boolean): void {
      reject = v;
    },
    /** Holds every request until the returned function is called. */
    hold(): () => void {
      let open!: () => void;
      gate = new Promise<void>((r) => (open = r));
      return () => {
        gate = null;
        open();
      };
    },
  };
}
