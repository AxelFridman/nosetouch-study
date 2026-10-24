/**
 * Audio / haptic / speech cues and the screen wake lock.
 *
 * Audio: Web Audio API oscillator beeps (no audio files). The AudioContext is created/resumed in
 * unlock(), which the UI calls inside the first user gesture (iOS requires it). Cue sounds:
 *   ready       : 440 Hz, 80 ms
 *   go          : 880 Hz, 160 ms (loud, unmistakable — it is the start signal)
 *   end         : two 660 Hz beeps of 90 ms separated by 70 ms
 *   error       : 220 Hz square-ish, 250 ms
 *   block_start : 523 Hz then 784 Hz, 100 ms each
 * Vibration: navigator.vibrate when available and enabled (Android; iOS has no Vibration API) — ONLY
 * for 'block_start'. Never for ready/go/end/error: those happen while recording, and the phone's own
 * vibration motor contaminates the accelerometer (a buzz at GO would land exactly on movement onset).
 * unlock() also sets navigator.audioSession.type = 'playback' when that API exists (Safari 16.4+;
 * keeps beeps audible when the iPhone ringer switch is on silent).
 * Speech: speechSynthesis with the language's voice (he-IL / en-US) when enabled; cancel() before
 * speaking; never throws. NOTE for callers: iOS only speaks when say() is called while a user
 * gesture is being processed — call it synchronously inside tap handlers, never from timers.
 * play() must never throw and must be cheap (called from sensor handlers).
 *
 * Implementation notes:
 *  - One AudioContext at a time; unlock() is idempotent and cheap, so the UI may call it in EVERY
 *    "start" tap (iOS suspends/interrupts the context after the page was hidden and only a gesture
 *    can resume it). unlock() does its work synchronously before returning the promise, so the
 *    gesture is still active when resume() runs.
 *  - iOS repair (WebKit bugs 276016, 273511, 283419): after a call, an alarm or an app switch a
 *    context can read 'interrupted' and stay silent, ignore resume(), or even read 'running' while
 *    silent. unlock() therefore runs suspend() and then resume() inside the tap whenever the context
 *    is not 'running', and on iPhones/iPads (WebKit) in EVERY tap; a context that is still not
 *    'running' REPAIR_CHECK_MS later is closed and replaced by the next tap.
 *  - play() never creates the context (that can take tens of ms — far too slow for a sensor
 *    handler) and skips the sound while the context is not 'running' (a suspended context would
 *    play the beep LATE, at an unknown time); it asks for a resume so the next cue can sound.
 *  - Tones are scheduled at ctx.currentTime with 5 ms attack / 15 ms release ramps (no clicks).
 *  - Voices load asynchronously in Chrome ('voiceschanged'); the voice list is re-read lazily.
 *    Android may report Hebrew as 'iw' and use '_' separators.
 *
 * WakeLockGuard: navigator.wakeLock.request('screen'); re-acquires on visibilitychange→visible
 * while active; release() stops re-acquiring. No-op where unsupported. `active` = a lock is
 * currently held. Safari may require a user gesture for the request, so callers should also call
 * acquire() inside taps (cheap when the lock is already held).
 *
 * IMPLEMENTED BY: sensors agent.
 */
import type { CueKind, CueSink, Language } from './types';

export interface CueOptions {
  audio: boolean;
  vibrate: boolean;
  speech: boolean;
}

export interface AppCueSink extends CueSink {
  /** Create/resume (or repair, see the module doc) the AudioContext. Call from a user gesture. */
  unlock(): Promise<void>;
  /** true when an utterance was queued (speech on, a voice engine, not silenced) */
  say(text: string, lang: Language): boolean;
  setOptions(o: CueOptions): void;
}

export interface CueSinkEnv {
  /**
   * Checked on every cue: while it returns true, play(), say() and unlock() make no sound, no
   * speech and no vibration, whatever the options say. The app uses it so a computer running the
   * simulator or an automated browser never talks or beeps.
   */
  silenced?: () => boolean;
}

export interface CueTone {
  freq: number; // Hz
  start: number; // s after the cue
  dur: number; // s
  type: 'sine' | 'square';
  gain: number; // peak, 0..1
}

/** The sound of every cue (exported for the About screen and tests). */
export const CUE_SOUNDS: Record<CueKind, CueTone[]> = {
  ready: [{ freq: 440, start: 0, dur: 0.08, type: 'sine', gain: 0.5 }],
  go: [{ freq: 880, start: 0, dur: 0.16, type: 'sine', gain: 0.95 }],
  end: [
    { freq: 660, start: 0, dur: 0.09, type: 'sine', gain: 0.6 },
    { freq: 660, start: 0.16, dur: 0.09, type: 'sine', gain: 0.6 },
  ],
  // A square wave carries ~3x the energy of a sine at the same peak: keep it quieter.
  error: [{ freq: 220, start: 0, dur: 0.25, type: 'square', gain: 0.25 }],
  block_start: [
    { freq: 523, start: 0, dur: 0.1, type: 'sine', gain: 0.6 },
    { freq: 784, start: 0.1, dur: 0.1, type: 'sine', gain: 0.6 },
  ],
};

/** Vibration pattern of 'block_start' (ms). Never used during a recording. */
export const BLOCK_START_VIBRATION: number[] = [80, 60, 80];

const ATTACK_S = 0.005;
const RELEASE_S = 0.015;

export const SPEECH_LANG: Record<Language, string> = { en: 'en-US', he: 'he-IL' };

// ───────────────────────────── loose views of the browser APIs ─────────────────────────────

interface ParamLike {
  setValueAtTime(value: number, time: number): unknown;
  linearRampToValueAtTime(value: number, time: number): unknown;
}

interface NodeLike {
  connect(dest: unknown): unknown;
  disconnect(): void;
}

interface OscillatorLike extends NodeLike {
  type: string;
  frequency: ParamLike;
  onended: (() => void) | null;
  start(when?: number): void;
  stop(when?: number): void;
}

interface GainLike extends NodeLike {
  gain: ParamLike;
}

interface BufferSourceLike extends NodeLike {
  buffer: unknown;
  start(when?: number): void;
}

interface AudioContextLike {
  readonly state: string;
  readonly currentTime: number;
  readonly destination: unknown;
  createOscillator(): OscillatorLike;
  createGain(): GainLike;
  createBuffer?(channels: number, length: number, sampleRate: number): unknown;
  createBufferSource?(): BufferSourceLike;
  resume(): Promise<void>;
  suspend?(): Promise<void>;
  close?(): Promise<void>;
}

type AudioContextCtor = new () => AudioContextLike;

interface VoiceLike {
  lang: string;
  name: string;
  default?: boolean;
  localService?: boolean;
}

interface UtteranceLike {
  lang: string;
  voice: VoiceLike | null;
  rate: number;
  pitch: number;
  volume: number;
}

interface SynthLike {
  speaking?: boolean;
  pending?: boolean;
  speak(u: UtteranceLike): void;
  cancel(): void;
  getVoices?(): VoiceLike[];
  addEventListener?(type: string, fn: () => void): void;
}

interface CueGlobals {
  AudioContext?: unknown;
  webkitAudioContext?: unknown;
  speechSynthesis?: SynthLike;
  SpeechSynthesisUtterance?: unknown;
  navigator?: {
    webdriver?: boolean;
    userAgent?: string;
    maxTouchPoints?: number;
    vibrate?: (pattern: number[]) => boolean;
    audioSession?: { type?: string };
    wakeLock?: { request?: (type: 'screen') => Promise<SentinelLike> } | null;
  };
  document?: {
    visibilityState?: string;
    addEventListener?(type: string, fn: () => void): void;
    removeEventListener?(type: string, fn: () => void): void;
  };
}

function G(): CueGlobals {
  return globalThis as unknown as CueGlobals;
}

const noop = (): void => undefined;

/** unlock(): a context that is not 'running' this long after its repair is replaced at the next tap. */
export const REPAIR_CHECK_MS = 300;

/** iPhone / iPad (every iOS browser is WebKit; iPadOS may report a Mac with touch). */
export function isTouchWebKit(): boolean {
  try {
    const nav = G().navigator;
    const ua = typeof nav?.userAgent === 'string' ? nav.userAgent : '';
    return /\b(iPhone|iPad|iPod)\b/.test(ua) || (/\bMacintosh\b/.test(ua) && (nav?.maxTouchPoints ?? 0) > 1);
  } catch {
    return false;
  }
}

/**
 * True in browsers driven by automation (Playwright, Puppeteer, Selenium all set
 * navigator.webdriver). Chrome on Windows speaks through the OS speech engine, so even a headless
 * or --mute-audio browser can talk out loud: the app itself must stay quiet there.
 */
export function isAutomatedBrowser(): boolean {
  try {
    return G().navigator?.webdriver === true;
  } catch {
    return false;
  }
}

// ───────────────────────────── voices ─────────────────────────────

function normTag(tag: string): string {
  return tag.replace(/_/g, '-').toLowerCase();
}

/** Best voice for `lang`: exact region > same language; then on-device > remote; then default. */
export function pickVoice<V extends VoiceLike>(voices: readonly V[], lang: Language): V | null {
  const exact = lang === 'he' ? ['he-il', 'iw-il'] : ['en-us'];
  const prefixes = lang === 'he' ? ['he', 'iw'] : ['en'];
  let best: V | null = null;
  let bestScore = 0;
  for (const v of voices) {
    if (!v || typeof v.lang !== 'string') continue;
    const tag = normTag(v.lang);
    let score = 0;
    if (exact.includes(tag)) score = 4;
    else if (prefixes.some((p) => tag === p || tag.startsWith(`${p}-`))) score = 2;
    if (score === 0) continue;
    if (v.localService) score += 1;
    if (v.default) score += 0.5;
    if (score > bestScore) {
      bestScore = score;
      best = v;
    }
  }
  return best;
}

// ───────────────────────────── cue sink ─────────────────────────────

export function createCueSink(o: CueOptions, env: CueSinkEnv = {}): AppCueSink {
  let opts: CueOptions = { ...o };
  let ctx: AudioContextLike | null = null;
  /** the context stayed silent after a repair: the next unlock() replaces it */
  let replaceCtx = false;
  const isSilenced = (): boolean => {
    try {
      return env.silenced?.() === true;
    } catch {
      return false;
    }
  };
  let voicesFor: SynthLike | null = null;
  let voices: VoiceLike[] = [];

  function refreshVoices(synth: SynthLike): void {
    try {
      voices = typeof synth.getVoices === 'function' ? (synth.getVoices() ?? []) : [];
    } catch {
      voices = [];
    }
  }

  function trackVoices(synth: SynthLike): void {
    if (voicesFor === synth) return;
    voicesFor = synth;
    refreshVoices(synth);
    try {
      synth.addEventListener?.('voiceschanged', () => refreshVoices(synth));
    } catch {
      // older engines: the lazy refresh in say() covers them
    }
  }

  // Starting to load voices early gives Chrome time to deliver them before the first say().
  try {
    const synth = G().speechSynthesis;
    if (synth) trackVoices(synth);
  } catch {
    // ignore
  }

  function scheduleTone(c: AudioContextLike, tone: CueTone, at: number): void {
    const t0 = at + tone.start;
    const t1 = t0 + tone.dur;
    const osc = c.createOscillator();
    const amp = c.createGain();
    osc.type = tone.type;
    osc.frequency.setValueAtTime(tone.freq, t0);
    amp.gain.setValueAtTime(0, t0);
    amp.gain.linearRampToValueAtTime(tone.gain, t0 + ATTACK_S);
    amp.gain.setValueAtTime(tone.gain, t1 - RELEASE_S);
    amp.gain.linearRampToValueAtTime(0, t1);
    osc.connect(amp);
    amp.connect(c.destination);
    osc.onended = () => {
      try {
        osc.disconnect();
        amp.disconnect();
      } catch {
        // already disconnected
      }
    };
    osc.start(t0);
    osc.stop(t1 + 0.005);
  }

  function unlock(): Promise<void> {
    if (isSilenced()) return Promise.resolve();
    // Synchronous part first: it must run while the user gesture is still being processed.
    const g = G();
    try {
      const session = g.navigator?.audioSession;
      if (session && 'type' in session) session.type = 'playback';
    } catch {
      // not settable on this engine
    }
    let resumed: Promise<void> = Promise.resolve();
    try {
      if (ctx && (ctx.state === 'closed' || replaceCtx)) {
        // a closed context can never play again; one that stayed silent after a repair is replaced
        const old = ctx;
        ctx = null;
        replaceCtx = false;
        if (old.state !== 'closed' && typeof old.close === 'function') {
          try {
            Promise.resolve(old.close()).then(noop, noop);
          } catch {
            // already closing
          }
        }
      }
      if (!ctx) {
        const Ctor = (g.AudioContext ?? g.webkitAudioContext) as AudioContextCtor | undefined;
        if (typeof Ctor === 'function') ctx = new Ctor();
      }
      const c = ctx;
      if (c) {
        // 'interrupted' anywhere, and every state on iPhones / iPads, get the repair cycle
        const repair = isTouchWebKit() || (c.state !== 'running' && c.state !== 'suspended');
        if (repair) {
          // 'interrupted' (or iOS silent while 'running'): suspend, then resume, both inside the tap
          if (typeof c.suspend === 'function') {
            try {
              Promise.resolve(c.suspend()).then(noop, noop);
            } catch {
              // not suspendable now: the resume below still runs
            }
          }
          resumed = Promise.resolve(c.resume()).then(noop, noop);
          setTimeout(() => {
            if (ctx === c && c.state !== 'running') replaceCtx = true;
          }, REPAIR_CHECK_MS);
        } else if (c.state !== 'running') {
          resumed = Promise.resolve(c.resume()).then(noop, noop);
        }
        // Old iOS needs a sound started inside the gesture to really unlock: one silent sample.
        if (typeof c.createBuffer === 'function' && typeof c.createBufferSource === 'function') {
          const src = c.createBufferSource();
          src.buffer = c.createBuffer(1, 1, 22050);
          src.connect(c.destination);
          src.start(0);
        }
      }
    } catch {
      // no Web Audio: cues stay silent, the app keeps working
    }
    return resumed;
  }

  function play(kind: CueKind): void {
    try {
      if (isSilenced()) return;
      if (kind === 'block_start' && opts.vibrate) {
        const nav = G().navigator;
        if (nav && typeof nav.vibrate === 'function') nav.vibrate(BLOCK_START_VIBRATION);
      }
      if (!opts.audio) return;
      const c = ctx;
      if (!c) return;
      if (c.state !== 'running') {
        // A suspended context would play this beep late; skip it and try to recover for the next.
        if (c.state !== 'closed') Promise.resolve(c.resume()).then(noop, noop);
        return;
      }
      const tones = CUE_SOUNDS[kind];
      if (!tones) return;
      const at = c.currentTime;
      for (const tone of tones) scheduleTone(c, tone, at);
    } catch {
      // cues are best-effort: never disturb the recording
    }
  }

  function say(text: string, lang: Language): boolean {
    try {
      if (!opts.speech || !text || isSilenced()) return false;
      const g = G();
      const synth = g.speechSynthesis;
      const Utterance = g.SpeechSynthesisUtterance as (new (text: string) => UtteranceLike) | undefined;
      if (!synth || typeof Utterance !== 'function') return false;
      trackVoices(synth);
      if (voices.length === 0) refreshVoices(synth);
      if (synth.speaking || synth.pending) synth.cancel();
      const u = new Utterance(text);
      u.lang = SPEECH_LANG[lang] ?? SPEECH_LANG.en;
      const voice = pickVoice(voices, lang);
      if (voice) u.voice = voice;
      u.rate = 1;
      u.pitch = 1;
      u.volume = 1;
      synth.speak(u);
      return true;
    } catch {
      // speech is optional
      return false;
    }
  }

  function setOptions(next: CueOptions): void {
    opts = { ...next };
    if (!opts.speech) {
      try {
        G().speechSynthesis?.cancel();
      } catch {
        // ignore
      }
    }
  }

  return { play, say, unlock, setOptions };
}

/** Cue sink that only records what was played (tests, offline synthesis). say() is recorded as
 * "say:<lang>:<text>" so cue-only assertions can filter it out. */
export function createRecordingCueSink(): CueSink & { played: string[] } {
  const played: string[] = [];
  return {
    played,
    play(kind: CueKind): void {
      played.push(kind);
    },
    say(text: string, lang: Language): void {
      played.push(`say:${lang}:${text}`);
    },
  };
}

// ───────────────────────────── wake lock ─────────────────────────────

interface SentinelLike {
  readonly released?: boolean;
  release(): Promise<void>;
  addEventListener?(type: 'release', fn: () => void): void;
  onrelease?: (() => void) | null;
}

export interface WakeLockGuard {
  acquire(): Promise<boolean>;
  release(): Promise<void>;
  readonly active: boolean;
}

export function createWakeLockGuard(): WakeLockGuard {
  let wanted = false;
  let sentinel: SentinelLike | null = null;
  let pending: Promise<boolean> | null = null;
  let listening = false;

  const held = (): boolean => sentinel !== null && sentinel.released !== true;

  function request(): Promise<boolean> {
    if (held()) return Promise.resolve(true);
    if (pending) return pending;
    let wakeLock: { request?: (type: 'screen') => Promise<SentinelLike> } | null | undefined;
    try {
      wakeLock = G().navigator?.wakeLock;
    } catch {
      wakeLock = null;
    }
    if (!wakeLock || typeof wakeLock.request !== 'function') return Promise.resolve(false);
    let req: Promise<SentinelLike>;
    try {
      req = Promise.resolve(wakeLock.request('screen')); // synchronous call: keeps the user gesture
    } catch {
      return Promise.resolve(false);
    }
    const p = req.then(
      (s) => {
        if (!wanted) {
          // release() ran while the request was in flight
          Promise.resolve(s.release()).catch(noop);
          return false;
        }
        sentinel = s;
        const onRelease = (): void => {
          if (sentinel === s) sentinel = null; // the system released it (page hidden, battery saver)
        };
        if (typeof s.addEventListener === 'function') s.addEventListener('release', onRelease);
        else s.onrelease = onRelease;
        return true;
      },
      () => false, // NotAllowedError (hidden page, no gesture on Safari, battery saver)
    );
    pending = p;
    void p.finally(() => {
      if (pending === p) pending = null;
    });
    return p;
  }

  const onVisibility = (): void => {
    if (wanted && G().document?.visibilityState === 'visible' && !held()) void request();
  };

  function listen(on: boolean): void {
    if (on === listening) return;
    try {
      const doc = G().document;
      if (on) doc?.addEventListener?.('visibilitychange', onVisibility);
      else doc?.removeEventListener?.('visibilitychange', onVisibility);
      listening = on;
    } catch {
      // no document (tests, workers)
    }
  }

  return {
    acquire(): Promise<boolean> {
      wanted = true;
      listen(true);
      return request();
    },
    async release(): Promise<void> {
      wanted = false;
      listen(false);
      const s = sentinel;
      sentinel = null;
      if (s && s.released !== true) {
        try {
          await s.release();
        } catch {
          // already released
        }
      }
    },
    get active(): boolean {
      return held();
    },
  };
}
