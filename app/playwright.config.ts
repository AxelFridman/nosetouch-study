import { defineConfig, devices } from '@playwright/test';

// End-to-end tests drive the real UI in simulation mode (?sim=1): the built-in simulator plays the
// participant, so the v2 participant path (welcome + consent → profile → session: How-to card, free
// trial, hand question, hands-free blocks, checkpoint → done) runs on a desktop browser with a phone
// viewport. ?sim=1&setup=1 opens the researchers' setup screen for short custom plans. Uses the dev
// server, so no service worker is involved. Every spec imports test/expect from e2e/fixtures.ts.
// Headless only: never run it headed or in UI mode on the researcher's computer.
export default defineConfig({
  testDir: './e2e',
  timeout: 240_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:5173/',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    acceptDownloads: true,
    // a test id that no longer exists fails within 15 s instead of waiting out the test timeout
    actionTimeout: 15_000,
    // Never make noise on the researcher's computer (also on a headed run). The app silences itself
    // under automation and in simulation, and e2e/fixtures.ts stubs speech and Web Audio, because
    // Chrome on Windows speaks through the operating system, which this flag does not mute.
    launchOptions: { args: ['--mute-audio'] },
  },
  projects: [
    {
      name: 'android-chromium',
      use: { ...devices['Pixel 7'] },
    },
  ],
  webServer: {
    command: 'npx vite --port 5173 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:5173/',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
