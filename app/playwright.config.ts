import { defineConfig, devices } from '@playwright/test';

// End-to-end tests drive the real UI in simulation mode (?sim=1): the built-in simulator plays a
// participant, so the full flow (consent → profile → sensors → calibration → session → export) runs
// on a desktop browser with a phone viewport. Uses the dev server so no service worker is involved.
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
