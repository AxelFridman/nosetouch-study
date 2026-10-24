import { defineConfig, devices } from '@playwright/test';
import type { NetworkGuardOptions } from './e2e/fixtures';

// Smoke test of a DEPLOYED build (production bundle + service worker on real HTTPS hosting):
//   npm run e2e:live                          (default: the GitHub Pages URL below)
//   LIVE_URL=https://<host>/<path>/ npm run e2e:live
// Runs the simulated-participant journeys of flow.spec.ts (?sim=1) and the boot check, so nothing
// touches real sensors. With uploads switched on in the deployed config.json, the simulated records
// ARE uploaded (marked simulated: true; the analysis drops them with rule S1): this config is the only
// one that lets the browser leave the test machine (externalRequests 'allow', see e2e/fixtures.ts).
// The local suite (playwright.config.ts) blocks every other host and serves config.json without uploads.
const LIVE_URL = process.env.LIVE_URL ?? 'https://axelfridman.github.io/nosetouch-study/';

export default defineConfig<NetworkGuardOptions>({
  testDir: './e2e',
  testMatch: ['flow.spec.ts', 'boot.spec.ts'],
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  retries: 1,
  reporter: [['list']],
  use: {
    baseURL: LIVE_URL.endsWith('/') ? LIVE_URL : `${LIVE_URL}/`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    acceptDownloads: true,
    actionTimeout: 15_000,
    // silent: see playwright.config.ts and e2e/fixtures.ts
    launchOptions: { args: ['--mute-audio'] },
    // the deployed build and its configured upload backend (see above)
    externalRequests: 'allow',
  },
  projects: [{ name: 'android-chromium', use: { ...devices['Pixel 7'] } }],
});
