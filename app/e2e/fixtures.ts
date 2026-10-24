/**
 * The `test` and `expect` of EVERY e2e spec (tests/e2eHarness.test.ts checks that each spec imports
 * them from here): the browser can never make a sound on the researcher's computer, and never reach
 * the live study backend.
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
 *
 * No data leaves the computer (option `externalRequests`, default 'block'): the committed
 * public/config.json uploads to the live Heroku backend, so every context
 *  - serves config.json with upload.mode 'none' (a spec that tests uploads sets its own same-origin
 *    endpoint in the researcher settings and answers it with page.route, which wins over this route);
 *  - aborts every request to a host other than the dev server (127.0.0.1 / localhost / the baseURL's
 *    host) and records it; the test FAILS at its end when anything was attempted.
 * Only the smoke test of a deployed build (playwright.live.config.ts) sets 'allow'.
 */
import { test as base, expect, type BrowserContext, type Page } from '@playwright/test';
import { guardedUrl, localHosts, offlineConfig } from './network';
import { silenceAudio, type SilenceLog } from './silence';

export type { SilenceLog };

export interface NetworkGuardOptions {
  /** 'block' (default): no request may leave the dev server; 'allow': the live smoke test only */
  externalRequests: 'block' | 'allow';
}

/** What the page tried to say or play so far (null when the silence stub is not in the page). */
export async function silenceLog(page: Page): Promise<SilenceLog | null> {
  return page.evaluate(() => {
    const log = (window as unknown as { __nosetouchSilence?: { spoken: string[]; audioContexts: number } }).__nosetouchSilence;
    return log ? { spoken: [...log.spoken], audioContexts: log.audioContexts } : null;
  });
}

/** Installs the network guard on a context; returns the list of blocked requests (method + URL). */
async function guardNetwork(context: BrowserContext, baseURL: string | undefined): Promise<string[]> {
  const local = localHosts(baseURL);
  const blocked: string[] = [];
  await context.route(
    (url) => guardedUrl(url, local),
    async (route) => {
      const req = route.request();
      const url = new URL(req.url());
      if (!local.has(url.hostname)) {
        blocked.push(`${req.method()} ${url.origin}${url.pathname}`);
        await route.abort('blockedbyclient');
        return;
      }
      const response = await route.fetch();
      let json: unknown;
      try {
        json = offlineConfig(await response.json());
      } catch {
        await route.fulfill({ response });
        return;
      }
      await route.fulfill({ response, json });
    },
  );
  return blocked;
}

export const test = base.extend<NetworkGuardOptions>({
  externalRequests: ['block', { option: true }],
  context: async ({ context, baseURL, externalRequests }, use) => {
    await context.addInitScript(silenceAudio);
    const blocked = externalRequests === 'block' ? await guardNetwork(context, baseURL) : null;
    await use(context);
    if (blocked) expect(blocked, 'requests to other hosts (the e2e must never reach the live backend)').toEqual([]);
  },
});

export { expect };
