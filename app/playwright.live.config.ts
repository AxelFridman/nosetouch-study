import { defineConfig, devices } from '@playwright/test';

// Smoke test of a DEPLOYED build (production bundle + service worker on real HTTPS hosting):
//   npm run e2e:live                          (default: the GitHub Pages URL below)
//   LIVE_URL=https://<host>/<path>/ npm run e2e:live
// Runs the simulated-participant journey (?sim=1), so nothing touches real sensors or the backend
// (the default config.json has uploads off; with uploads on, simulated records would be uploaded
// and are marked simulated: true).
const LIVE_URL = process.env.LIVE_URL ?? 'https://axelfridman.github.io/nosetouch-study/';

export default defineConfig({
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
  },
  projects: [{ name: 'android-chromium', use: { ...devices['Pixel 7'] } }],
});
