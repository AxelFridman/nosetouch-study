/**
 * core/cues.ts: Web Audio beeps, the iOS audio session, vibration (block_start only), speech
 * synthesis, the recording cue sink and the screen wake lock guard. Browser APIs are the fakes of
 * tests/helpers/sensorsFakes.ts, installed with vi.stubGlobal.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BLOCK_START_VIBRATION,
  CUE_SOUNDS,
  SPEECH_LANG,
  createCueSink,
  createRecordingCueSink,
  createWakeLockGuard,
  pickVoice,
  type CueOptions,
} from '../src/core/cues';
import type { CueKind } from '../src/core/types';
import { FakeDocument, fakeAudioContextClass, fakeSpeech, fakeWakeLock, type FakeVoice } from './helpers/sensorsFakes';

const KINDS: CueKind[] = ['ready', 'go', 'end', 'error', 'block_start'];
const ALL_ON: CueOptions = { audio: true, vibrate: true, speech: true };
const ATTACK_S = 0.005;
const RELEASE_S = 0.015;

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Installs a fake AudioContext (and an empty navigator unless one is given). */
function withAudio(opts: Parameters<typeof fakeAudioContextClass>[0] = {}, nav: Record<string, unknown> = {}) {
  const audio = fakeAudioContextClass(opts);
  vi.stubGlobal('AudioContext', audio.AudioContext);
  vi.stubGlobal('navigator', nav);
  return audio;
}

type Ctx = ReturnType<typeof fakeAudioContextClass>['instances'][number];

/** The tones scheduled on `ctx` from oscillator `from` on, relative to the context's currentTime. */
function scheduled(ctx: Ctx, from = 0) {
  return ctx.oscillators.slice(from).map((osc, i) => {
    const amp = ctx.gains[from + i];
    const [, freq, freqAt] = osc.frequency.events[0];
    return {
      osc,
      amp,
      type: osc.type,
      freq,
      freqAt: freqAt - ctx.currentTime,
      start: (osc.startAt ?? NaN) - ctx.currentTime,
      stop: (osc.stopAt ?? NaN) - ctx.currentTime,
      peak: Math.max(...amp.gain.events.map(([, v]) => v)),
    };
  });
}

// ───────────────────────────── the cue sounds ─────────────────────────────

describe('CUE_SOUNDS', () => {
  it('every cue kind has short, audible tones; GO is the loudest and highest', () => {
    for (const kind of KINDS) {
      const tones = CUE_SOUNDS[kind];
      expect(tones.length, kind).toBeGreaterThan(0);
      for (const t of tones) {
        expect(t.freq).toBeGreaterThan(100);
        expect(t.dur).toBeGreaterThan(2 * (ATTACK_S + RELEASE_S));
        expect(t.gain).toBeGreaterThan(0);
        expect(t.gain).toBeLessThanOrEqual(1);
        expect(t.start + t.dur, kind).toBeLessThanOrEqual(0.3); // cues must not run into the movement
      }
    }
    const peak = (k: CueKind) => Math.max(...CUE_SOUNDS[k].map((t) => t.gain));
    for (const k of KINDS.filter((k) => k !== 'go')) expect(peak('go')).toBeGreaterThan(peak(k));
    expect(Math.max(...CUE_SOUNDS.go.map((t) => t.freq))).toBe(Math.max(...KINDS.flatMap((k) => CUE_SOUNDS[k].map((t) => t.freq))));
  });

  it('the end cue is a double beep with a gap, block_start a rising pair', () => {
    const [a, b] = CUE_SOUNDS.end;
    expect(CUE_SOUNDS.end).toHaveLength(2);
    expect(b.start).toBeGreaterThan(a.start + a.dur);
    expect(CUE_SOUNDS.block_start.map((t) => t.freq)).toEqual([523, 784]);
    expect(BLOCK_START_VIBRATION).toEqual([80, 60, 80]);
    expect(SPEECH_LANG).toEqual({ en: 'en-US', he: 'he-IL' });
  });
});

// ───────────────────────────── audio ─────────────────────────────

describe('createCueSink: audio', () => {
  it('play() before unlock() is silent: it never creates an AudioContext (too slow for a sensor handler)', () => {
    const audio = withAudio();
    const sink = createCueSink(ALL_ON);
    for (const k of KINDS) expect(() => sink.play(k)).not.toThrow();
    expect(audio.instances).toHaveLength(0);
  });

  it('unlock() creates the context, resumes it and starts a silent sample (old iOS needs one in the gesture)', async () => {
    const audio = withAudio({ initialState: 'suspended' });
    const sink = createCueSink(ALL_ON);
    const p = sink.unlock();
    // the synchronous part ran before unlock() returned, i.e. inside the user gesture
    expect(audio.instances).toHaveLength(1);
    expect(audio.instances[0].resumeCalls).toBe(1);
    expect(audio.instances[0].silentStarts).toBe(1);
    await expect(p).resolves.toBeUndefined();
    expect(audio.instances[0].state).toBe('running');
  });

  it('unlock() is idempotent and cheap: one context for the app, no resume while running', async () => {
    const audio = withAudio({ initialState: 'suspended' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    await sink.unlock();
    await sink.unlock();
    expect(audio.instances).toHaveLength(1);
    expect(audio.instances[0].resumeCalls).toBe(1);
  });

  it('unlock() replaces a closed context (a closed context can never play again)', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    audio.instances[0].state = 'closed';
    await sink.unlock();
    expect(audio.instances).toHaveLength(2);
    sink.play('go');
    expect(audio.instances[0].oscillators).toHaveLength(0);
    expect(audio.instances[1].oscillators).toHaveLength(1);
  });

  it('falls back to webkitAudioContext (older Safari)', async () => {
    const audio = fakeAudioContextClass({ initialState: 'running' });
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('webkitAudioContext', audio.AudioContext);
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    sink.play('ready');
    expect(audio.instances).toHaveLength(1);
    expect(audio.instances[0].oscillators).toHaveLength(1);
  });

  it('unlock() and play() never throw when the AudioContext cannot be constructed', async () => {
    const audio = withAudio({ throwOnConstruct: true });
    const sink = createCueSink(ALL_ON);
    await expect(sink.unlock()).resolves.toBeUndefined();
    for (const k of KINDS) expect(() => sink.play(k)).not.toThrow();
    expect(audio.instances).toHaveLength(0);
  });

  it('unlock() resolves even when resume() is refused', async () => {
    const audio = fakeAudioContextClass({ initialState: 'suspended' });
    class Refusing extends audio.AudioContext {
      override resume(): Promise<void> {
        this.resumeCalls++;
        return Promise.reject(new DOMException('not allowed', 'NotAllowedError'));
      }
    }
    vi.stubGlobal('AudioContext', Refusing);
    vi.stubGlobal('navigator', {});
    const sink = createCueSink(ALL_ON);
    await expect(sink.unlock()).resolves.toBeUndefined();
    expect(() => sink.play('go')).not.toThrow(); // still suspended: skipped, resume asked again
    await flush();
    expect(audio.instances[0].resumeCalls).toBe(2);
    expect(audio.instances[0].oscillators).toHaveLength(0);
  });

  it('play() schedules exactly the tones of CUE_SOUNDS at currentTime, with click-free ramps', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    const ctx = audio.instances[0];
    for (const kind of KINDS) {
      const from = ctx.oscillators.length;
      sink.play(kind);
      const tones = scheduled(ctx, from);
      expect(tones.map((t) => [t.type, t.freq]), kind).toEqual(CUE_SOUNDS[kind].map((t) => [t.type, t.freq]));
      tones.forEach((t, i) => {
        const spec = CUE_SOUNDS[kind][i];
        const t0 = ctx.currentTime + spec.start;
        const t1 = t0 + spec.dur;
        expect(t.start).toBeCloseTo(spec.start, 9);
        expect(t.freqAt).toBeCloseTo(spec.start, 9);
        expect(t.stop).toBeCloseTo(spec.start + spec.dur + 0.005, 9);
        expect(t.peak).toBeCloseTo(spec.gain, 9);
        expect(t.amp.gain.events).toEqual([
          ['set', 0, t0],
          ['ramp', spec.gain, t0 + ATTACK_S],
          ['set', spec.gain, t1 - RELEASE_S],
          ['ramp', 0, t1],
        ]);
        // oscillator → its own gain → speakers
        expect(t.osc.connections).toHaveLength(1);
        expect(t.osc.connections[0]).toBe(t.amp);
        expect(t.amp.connections).toHaveLength(1);
        expect(t.amp.connections[0]).toBe(ctx.destination);
      });
    }
  });

  it('a finished tone disconnects itself, and a failing disconnect is swallowed', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    sink.play('end');
    const [first, second] = audio.instances[0].oscillators;
    const spy = vi.spyOn(first, 'disconnect');
    first.onended?.();
    expect(spy).toHaveBeenCalledOnce();
    vi.spyOn(second, 'disconnect').mockImplementation(() => {
      throw new Error('already disconnected');
    });
    expect(() => second.onended?.()).not.toThrow();
  });

  it('a suspended context skips the beep (it would sound late) and asks for a resume', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    const ctx = audio.instances[0];
    ctx.state = 'suspended'; // iOS after the page was hidden, or an interruption
    sink.play('go');
    expect(ctx.oscillators).toHaveLength(0);
    expect(ctx.resumeCalls).toBe(1);
    await flush();
    sink.play('go'); // resumed: the next cue sounds
    expect(ctx.oscillators).toHaveLength(1);
  });

  it('a closed context neither plays nor tries to resume', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    const ctx = audio.instances[0];
    ctx.state = 'closed';
    sink.play('go');
    expect(ctx.oscillators).toHaveLength(0);
    expect(ctx.resumeCalls).toBe(0);
  });

  it('play() never throws when building the audio graph fails, nor for an unknown cue', async () => {
    withAudio({ initialState: 'running', throwOnCreate: true });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    for (const k of KINDS) expect(() => sink.play(k)).not.toThrow();
    expect(() => sink.play('nope' as CueKind)).not.toThrow();
  });

  it('setOptions({audio: false}) silences play(); switching it back on restores it', async () => {
    const audio = withAudio({ initialState: 'running' });
    const sink = createCueSink(ALL_ON);
    await sink.unlock();
    sink.setOptions({ ...ALL_ON, audio: false });
    sink.play('go');
    expect(audio.instances[0].oscillators).toHaveLength(0);
    sink.setOptions(ALL_ON);
    sink.play('go');
    expect(audio.instances[0].oscillators).toHaveLength(1);
  });

  it('setOptions() copies the options (later mutation of the caller object has no effect)', async () => {
    const audio = withAudio({ initialState: 'running' });
    const opts: CueOptions = { ...ALL_ON };
    const sink = createCueSink(opts);
    await sink.unlock();
    opts.audio = false;
    sink.play('ready');
    expect(audio.instances[0].oscillators).toHaveLength(1);
  });
});

// ───────────────────────────── iOS audio session ─────────────────────────────

describe('createCueSink: navigator.audioSession (beeps despite the iPhone silent switch)', () => {
  it("unlock() sets audioSession.type = 'playback' when the API exists", async () => {
    const session = { type: 'auto' };
    withAudio({}, { audioSession: session });
    await createCueSink(ALL_ON).unlock();
    expect(session.type).toBe('playback');
  });

  it('works without the API, and without Web Audio at all', async () => {
    vi.stubGlobal('navigator', {});
    await expect(createCueSink(ALL_ON).unlock()).resolves.toBeUndefined();
    const session = { type: 'auto' };
    vi.stubGlobal('navigator', { audioSession: session }); // no AudioContext global in Node
    await createCueSink(ALL_ON).unlock();
    expect(session.type).toBe('playback');
  });

  it('a setter that throws is ignored and the context is still created', async () => {
    const session = {};
    Object.defineProperty(session, 'type', {
      get: () => 'auto',
      set: () => {
        throw new TypeError('read-only');
      },
      enumerable: true,
    });
    const audio = withAudio({}, { audioSession: session });
    await expect(createCueSink(ALL_ON).unlock()).resolves.toBeUndefined();
    expect(audio.instances).toHaveLength(1);
  });
});

// ───────────────────────────── vibration ─────────────────────────────

describe('createCueSink: vibration', () => {
  it('only block_start vibrates: a buzz during a recording would contaminate the accelerometer', () => {
    const vibrate = vi.fn(() => true);
    withAudio({}, { vibrate });
    const sink = createCueSink(ALL_ON);
    for (const k of ['ready', 'go', 'end', 'error'] as CueKind[]) sink.play(k);
    expect(vibrate).not.toHaveBeenCalled();
    sink.play('block_start');
    expect(vibrate).toHaveBeenCalledOnce();
    expect(vibrate).toHaveBeenCalledWith(BLOCK_START_VIBRATION);
  });

  it('vibrates before unlock() and with audio off; not at all when vibration is off', () => {
    const vibrate = vi.fn(() => true);
    withAudio({}, { vibrate });
    const sink = createCueSink({ audio: false, vibrate: true, speech: false });
    sink.play('block_start');
    expect(vibrate).toHaveBeenCalledOnce();
    sink.setOptions({ audio: false, vibrate: false, speech: false });
    sink.play('block_start');
    expect(vibrate).toHaveBeenCalledOnce();
  });

  it('no Vibration API (iOS) or a throwing one: no error', () => {
    withAudio({}, {});
    expect(() => createCueSink(ALL_ON).play('block_start')).not.toThrow();
    withAudio({}, {
      vibrate: () => {
        throw new Error('blocked');
      },
    });
    expect(() => createCueSink(ALL_ON).play('block_start')).not.toThrow();
  });
});

// ───────────────────────────── speech ─────────────────────────────

const VOICES: FakeVoice[] = [
  { name: 'Samantha', lang: 'en-US', localService: true },
  { name: 'Daniel', lang: 'en-GB', localService: true },
  { name: 'Carmit (network)', lang: 'he-IL', localService: false },
  { name: 'Carmit', lang: 'he-IL', localService: true },
];

function withSpeech(voices: FakeVoice[] = VOICES, opts: { asyncVoices?: boolean } = {}) {
  const speech = fakeSpeech(voices, opts);
  vi.stubGlobal('navigator', {});
  vi.stubGlobal('speechSynthesis', speech.synth);
  vi.stubGlobal('SpeechSynthesisUtterance', speech.Utterance);
  return speech;
}

describe('createCueSink: speech', () => {
  it('Hebrew: he-IL with the best Hebrew voice (on-device beats network)', () => {
    const speech = withSpeech();
    createCueSink(ALL_ON).say?.('שלום', 'he');
    expect(speech.spoken).toHaveLength(1);
    expect(speech.spoken[0]).toMatchObject({ text: 'שלום', lang: 'he-IL', rate: 1, pitch: 1, volume: 1 });
    expect(speech.spoken[0].voice?.name).toBe('Carmit');
  });

  it('English: en-US with an en-US voice', () => {
    const speech = withSpeech();
    createCueSink(ALL_ON).say?.('Ready', 'en');
    expect(speech.spoken[0]).toMatchObject({ lang: 'en-US' });
    expect(speech.spoken[0].voice?.name).toBe('Samantha');
  });

  it("Android's 'iw_IL' tag counts as Hebrew", () => {
    const speech = withSpeech([{ name: 'Google עברית', lang: 'iw_IL' }]);
    createCueSink(ALL_ON).say?.('שלום', 'he');
    expect(speech.spoken[0].voice?.name).toBe('Google עברית');
    expect(speech.spoken[0].lang).toBe('he-IL');
  });

  it('no voice for the language: the lang is still set and the voice left to the browser', () => {
    const speech = withSpeech([{ name: 'Thomas', lang: 'fr-FR' }]);
    createCueSink(ALL_ON).say?.('שלום', 'he');
    expect(speech.spoken[0]).toMatchObject({ lang: 'he-IL', voice: null });
  });

  it("voices that arrive later (Chrome's voiceschanged) are used", () => {
    const speech = withSpeech(VOICES, { asyncVoices: true });
    const sink = createCueSink(ALL_ON);
    sink.say?.('Ready', 'en');
    expect(speech.spoken[0].voice).toBeNull();
    speech.loadVoices();
    sink.say?.('Ready', 'en');
    expect(speech.spoken[1].voice?.name).toBe('Samantha');
  });

  it('cancels whatever is still speaking or queued before speaking', () => {
    const speech = withSpeech();
    const sink = createCueSink(ALL_ON);
    sink.say?.('one', 'en');
    expect(speech.synth.cancel).not.toHaveBeenCalled();
    speech.synth.speaking = true;
    sink.say?.('two', 'en');
    expect(speech.synth.cancel).toHaveBeenCalledOnce();
    expect(speech.synth.cancel.mock.invocationCallOrder[0]).toBeLessThan(speech.synth.speak.mock.invocationCallOrder[1]);
    speech.synth.speaking = false;
    speech.synth.pending = true;
    sink.say?.('three', 'en');
    expect(speech.synth.cancel).toHaveBeenCalledTimes(2);
  });

  it('speech off or empty text: nothing is spoken', () => {
    const speech = withSpeech();
    createCueSink({ ...ALL_ON, speech: false }).say?.('Ready', 'en');
    createCueSink(ALL_ON).say?.('', 'en');
    expect(speech.synth.speak).not.toHaveBeenCalled();
  });

  it('setOptions({speech: false}) stops speech in progress; switching back on speaks again', () => {
    const speech = withSpeech();
    const sink = createCueSink(ALL_ON);
    sink.setOptions({ ...ALL_ON, speech: false });
    expect(speech.synth.cancel).toHaveBeenCalledOnce();
    sink.say?.('Ready', 'en');
    expect(speech.spoken).toHaveLength(0);
    sink.setOptions(ALL_ON);
    sink.say?.('Ready', 'en');
    expect(speech.spoken).toHaveLength(1);
  });

  it('never throws: no API, a throwing speak(), constructor or getVoices()', () => {
    vi.stubGlobal('navigator', {});
    expect(() => createCueSink(ALL_ON).say?.('Ready', 'en')).not.toThrow(); // no speechSynthesis in Node

    const speech = withSpeech();
    speech.synth.speak.mockImplementation(() => {
      throw new Error('synthesis-failed');
    });
    expect(() => createCueSink(ALL_ON).say?.('Ready', 'en')).not.toThrow();

    withSpeech();
    vi.stubGlobal('SpeechSynthesisUtterance', function Broken() {
      throw new Error('no utterances here');
    });
    expect(() => createCueSink(ALL_ON).say?.('Ready', 'en')).not.toThrow();

    const flaky = withSpeech();
    flaky.synth.getVoices.mockImplementation(() => {
      throw new Error('voices unavailable');
    });
    const sink = createCueSink(ALL_ON);
    expect(() => sink.say?.('Ready', 'en')).not.toThrow();
    expect(flaky.spoken[0]).toMatchObject({ lang: 'en-US', voice: null });
    expect(() => sink.setOptions({ ...ALL_ON, speech: false })).not.toThrow();
  });
});

describe('pickVoice', () => {
  const v = (lang: string, extra: Partial<FakeVoice> = {}): FakeVoice => ({ name: lang, lang, ...extra });

  it('an exact region beats the bare language, on-device beats network, default breaks ties', () => {
    expect(pickVoice([v('en-GB', { localService: true }), v('en-US')], 'en')?.lang).toBe('en-US');
    expect(pickVoice([v('en-US', { default: true }), v('en-US', { name: 'local', localService: true })], 'en')?.name).toBe('local');
    expect(pickVoice([v('he-IL', { name: 'a' }), v('he-IL', { name: 'b', default: true })], 'he')?.name).toBe('b');
    expect(pickVoice([v('he')], 'he')?.lang).toBe('he');
  });

  it('normalises case and underscores; no false prefix matches; skips malformed entries', () => {
    expect(pickVoice([v('EN_us')], 'en')?.lang).toBe('EN_us');
    expect(pickVoice([v('hebrew'), v('ens-US')], 'he')).toBeNull();
    expect(pickVoice([v('ens-US')], 'en')).toBeNull();
    const malformed = [null, { name: 'x', lang: 42 }, v('he-IL')] as unknown as FakeVoice[];
    expect(pickVoice(malformed, 'he')?.lang).toBe('he-IL');
    expect(pickVoice([], 'en')).toBeNull();
  });
});

// ───────────────────────────── recording sink ─────────────────────────────

describe('createRecordingCueSink', () => {
  it('records cues and speech in order, speech as say:<lang>:<text>', () => {
    const sink = createRecordingCueSink();
    sink.play('ready');
    sink.say?.('Hold still', 'en');
    sink.play('go');
    sink.say?.('שלום', 'he');
    expect(sink.played).toEqual(['ready', 'say:en:Hold still', 'go', 'say:he:שלום']);
    expect(sink.played.filter((p) => !p.startsWith('say:'))).toEqual(['ready', 'go']);
  });
});

// ───────────────────────────── wake lock ─────────────────────────────

function withWakeLock(opts: { reject?: boolean } = {}) {
  const wl = fakeWakeLock(opts);
  const doc = new FakeDocument();
  vi.stubGlobal('navigator', { wakeLock: wl.wakeLock });
  vi.stubGlobal('document', doc);
  return { ...wl, doc };
}

describe('createWakeLockGuard', () => {
  it("acquire() requests a 'screen' lock and reports it as active", async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    expect(guard.active).toBe(false);
    await expect(guard.acquire()).resolves.toBe(true);
    expect(wl.wakeLock.request).toHaveBeenCalledWith('screen');
    expect(guard.active).toBe(true);
  });

  it('acquire() while a lock is held is cheap: no second request (callers repeat it in every tap)', async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    await guard.acquire();
    await expect(guard.acquire()).resolves.toBe(true);
    expect(wl.wakeLock.request).toHaveBeenCalledOnce();
  });

  it('concurrent acquire() calls share one request', async () => {
    const wl = withWakeLock();
    const open = wl.hold();
    const guard = createWakeLockGuard();
    const a = guard.acquire();
    const b = guard.acquire();
    open();
    await expect(Promise.all([a, b])).resolves.toEqual([true, true]);
    expect(wl.wakeLock.request).toHaveBeenCalledOnce();
    expect(wl.sentinels).toHaveLength(1);
  });

  it('release() releases the sentinel and stops listening for visibility changes', async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    await guard.acquire();
    expect(wl.doc.listenerCount('visibilitychange')).toBe(1);
    await guard.release();
    expect(wl.sentinels[0].release).toHaveBeenCalledOnce();
    expect(guard.active).toBe(false);
    expect(wl.doc.listenerCount('visibilitychange')).toBe(0);
    await expect(guard.release()).resolves.toBeUndefined(); // twice is harmless
  });

  it('re-acquires when the page becomes visible again after the system dropped the lock', async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    await guard.acquire();
    wl.doc.setVisibility('hidden');
    wl.sentinels[0].systemRelease(); // browsers release screen locks of hidden pages
    expect(guard.active).toBe(false);
    await flush();
    expect(wl.wakeLock.request).toHaveBeenCalledOnce(); // not while hidden
    wl.doc.setVisibility('visible');
    await flush();
    expect(wl.wakeLock.request).toHaveBeenCalledTimes(2);
    expect(guard.active).toBe(true);
    expect(wl.sentinels).toHaveLength(2);
  });

  it('a visible page that still holds its lock does not request another one', async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    await guard.acquire();
    wl.doc.setVisibility('visible');
    await flush();
    expect(wl.wakeLock.request).toHaveBeenCalledOnce();
  });

  it('after release() a visibility change does not re-acquire', async () => {
    const wl = withWakeLock();
    const guard = createWakeLockGuard();
    await guard.acquire();
    await guard.release();
    wl.doc.setVisibility('hidden');
    wl.doc.setVisibility('visible');
    await flush();
    expect(wl.wakeLock.request).toHaveBeenCalledOnce();
    expect(guard.active).toBe(false);
  });

  it('release() during an in-flight request releases the late sentinel', async () => {
    const wl = withWakeLock();
    const open = wl.hold();
    const guard = createWakeLockGuard();
    const pending = guard.acquire();
    await guard.release();
    open();
    await expect(pending).resolves.toBe(false);
    await flush();
    expect(wl.sentinels[0].release).toHaveBeenCalledOnce();
    expect(guard.active).toBe(false);
  });

  it('a refused request (NotAllowedError: hidden page, no gesture, battery saver) resolves false; a later one can succeed', async () => {
    const wl = withWakeLock({ reject: true });
    const guard = createWakeLockGuard();
    await expect(guard.acquire()).resolves.toBe(false);
    expect(guard.active).toBe(false);
    wl.setReject(false);
    await expect(guard.acquire()).resolves.toBe(true);
    expect(guard.active).toBe(true);
  });

  it('unsupported browsers: acquire() resolves false and release() is harmless', async () => {
    vi.stubGlobal('navigator', {});
    vi.stubGlobal('document', new FakeDocument());
    const guard = createWakeLockGuard();
    await expect(guard.acquire()).resolves.toBe(false);
    await expect(guard.release()).resolves.toBeUndefined();
    vi.stubGlobal('navigator', {
      wakeLock: {
        request: () => {
          throw new TypeError('not allowed by permissions policy');
        },
      },
    });
    await expect(createWakeLockGuard().acquire()).resolves.toBe(false);
  });

  it('works without a document (tests, workers)', async () => {
    const wl = fakeWakeLock();
    vi.stubGlobal('navigator', { wakeLock: wl.wakeLock });
    vi.stubGlobal('document', undefined);
    const guard = createWakeLockGuard();
    await expect(guard.acquire()).resolves.toBe(true);
    await expect(guard.release()).resolves.toBeUndefined();
  });

  it('a sentinel without addEventListener is tracked through onrelease', async () => {
    const sentinel = {
      released: false,
      onrelease: null as (() => void) | null,
      release: vi.fn(async () => undefined),
    };
    vi.stubGlobal('navigator', { wakeLock: { request: vi.fn(async () => sentinel) } });
    vi.stubGlobal('document', new FakeDocument());
    const guard = createWakeLockGuard();
    await guard.acquire();
    expect(typeof sentinel.onrelease).toBe('function');
    sentinel.onrelease?.(); // the system released it
    expect(guard.active).toBe(false);
  });
});
