/**
 * The in-page half of the e2e silence (see e2e/fixtures.ts): no Playwright import here, so the unit
 * test tests/e2eHarness.test.ts can run silenceAudio() against a fake window.
 */

export interface SilenceLog {
  /** text of every utterance the page asked speechSynthesis to speak */
  spoken: string[];
  /** attempts to construct an AudioContext / webkitAudioContext */
  audioContexts: number;
}

/**
 * Runs in every page and frame before its own scripts (context.addInitScript serialises it, so it
 * must stay self-contained: no imports, no closures over module variables). Replaces speechSynthesis
 * with a recorder that never speaks, and AudioContext / webkitAudioContext with a constructor that
 * throws (the app treats that as "no Web Audio"); window.__nosetouchSilence is what the page tried.
 */
export function silenceAudio(): void {
  const w = window as unknown as Record<string, unknown>;
  const log = { spoken: [] as string[], audioContexts: 0 };
  try {
    Object.defineProperty(w, '__nosetouchSilence', { configurable: true, value: log });
  } catch {
    w.__nosetouchSilence = log;
  }
  const synth = {
    speaking: false,
    pending: false,
    paused: false,
    onvoiceschanged: null,
    speak(u: { text?: unknown } | null | undefined) {
      log.spoken.push(String(u?.text ?? ''));
    },
    cancel() {},
    pause() {},
    resume() {},
    getVoices: () => [],
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => false,
  };
  try {
    Object.defineProperty(w, 'speechSynthesis', { configurable: true, get: () => synth });
  } catch {
    // not configurable here: --mute-audio and the app's own silence still hold
  }
  class NoAudioContext {
    constructor() {
      log.audioContexts += 1;
      throw new DOMException('audio is disabled in the e2e tests', 'NotSupportedError');
    }
  }
  for (const key of ['AudioContext', 'webkitAudioContext']) {
    try {
      Object.defineProperty(w, key, { configurable: true, writable: true, value: NoAudioContext });
    } catch {
      // see above
    }
  }
}
