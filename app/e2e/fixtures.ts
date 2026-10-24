/**
 * The `test` and `expect` of EVERY e2e spec (tests/e2eHarness.test.ts checks that each spec imports
 * them from here): the browser can never make a sound on the researcher's computer.
 *
 * Three layers keep a run silent:
 *  1. the app itself: in simulation mode (?sim=1) and in automated browsers (navigator.webdriver) its
 *     cue sink never creates an AudioContext and never speaks (core/cues.ts, app/services.ts);
 *  2. the Playwright configs launch Chromium with --mute-audio (also for headed runs);
 *  3. this fixture installs silenceAudio (e2e/silence.ts) in every page and frame BEFORE any app
 *     script runs: speech is recorded, never spoken (Chrome on Windows speaks through the operating
 *     system, which --mute-audio does not silence), and an AudioContext cannot even be created.
 * What the page tried is in window.__nosetouchSilence; the journeys assert that it stays
 * { spoken: [], audioContexts: 0 }, which proves layer 1 on its own.
 */
import { test as base, expect, type Page } from '@playwright/test';
import { silenceAudio, type SilenceLog } from './silence';

export type { SilenceLog };

/** What the page tried to say or play so far (null when the silence stub is not in the page). */
export async function silenceLog(page: Page): Promise<SilenceLog | null> {
  return page.evaluate(() => {
    const log = (window as unknown as { __nosetouchSilence?: { spoken: string[]; audioContexts: number } }).__nosetouchSilence;
    return log ? { spoken: [...log.spoken], audioContexts: log.audioContexts } : null;
  });
}

export const test = base.extend({
  context: async ({ context }, use) => {
    await context.addInitScript(silenceAudio);
    await use(context);
  },
});

export { expect };
