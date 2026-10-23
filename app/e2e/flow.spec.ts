/**
 * End-to-end participant journey in simulation mode (?sim=1): the built-in simulator plays the
 * participant, so the real UI runs welcome → consent → profile → sensors → calibration → setup →
 * session (2 trials) → done → export on a desktop browser with a phone viewport.
 * Only data-testid selectors are used (never UI text), so the flow is language-independent.
 * Run: npm run e2e (Playwright starts the Vite dev server, see playwright.config.ts).
 */
import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import type { ExportBundle } from '../src/core/types';
import { chooseTwoTrialProtocol, disableWebShare, fillProfile, inputOf, runCalibrationStep, runTrial, tapFresh } from './helpers';

test('simulated participant: full journey ends with a valid export', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await disableWebShare(page);

  await page.goto('./?sim=1&lang=en');

  // welcome → consent
  await page.getByTestId('btn-start').click();
  const consent = page.getByTestId('chk-consent');
  await consent.click();
  await expect(await inputOf(consent)).toBeChecked();
  await page.getByTestId('btn-consent-accept').click();

  // profile
  await fillProfile(page);
  await page.getByTestId('btn-save-profile').click();

  // sensor check (simulator)
  await page.getByTestId('btn-request-permission').click();
  const sensorsContinue = page.getByTestId('btn-continue');
  await expect(sensorsContinue).toBeEnabled({ timeout: 30_000 });
  await sensorsContinue.click();

  // calibration
  await runCalibrationStep(page, 'btn-calibrate-flat');
  await runCalibrationStep(page, 'btn-calibrate-upright');
  await page.getByTestId('btn-continue').click();

  // setup → session: one block per hand, one trial each
  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));

  const nextBlock = page.getByTestId('btn-next-block');
  const doneScreen = page.getByTestId('screen-done');
  for (let trial = 0; trial < 2; trial++) {
    await runTrial(page);
    await expect(
      nextBlock.or(page.getByTestId('btn-block-start')).or(page.getByTestId('btn-ready')).or(doneScreen).first(),
    ).toBeVisible({ timeout: 30_000 });
    if (await nextBlock.isVisible()) await tapFresh(page, nextBlock);
  }

  // done → export (download)
  await expect(doneScreen).toBeVisible({ timeout: 30_000 });
  const downloadPromise = page.waitForEvent('download');
  await page.getByTestId('btn-export').click();
  const download = await downloadPromise;
  const bundle = JSON.parse(await readFile(await download.path(), 'utf8')) as ExportBundle;

  expect(bundle.format).toBe('nose-touch-export');
  const good = bundle.trials.filter((t) => t.outcome.accepted && t.outcome.status === 'complete');
  expect(good).toHaveLength(2);
  for (const t of good) expect(t.streams.motion.t.length).toBeGreaterThan(50);
  expect(new Set(good.map((t) => t.condition.hand))).toEqual(new Set(['left', 'right']));
  expect(bundle.trials.length).toBeGreaterThan(0);
  expect(bundle.trials.every((t) => t.simulated === true)).toBe(true);
  expect(bundle.sessions).toHaveLength(1);
  expect(bundle.sessions[0].status).toBe('completed');
  expect(bundle.sessions[0].simulated).toBe(true);
  expect(bundle.sessions[0].progress.n_accepted).toBe(2);
  expect(pageErrors, `uncaught errors in the page:\n${pageErrors.join('\n')}`).toEqual([]);
});

test('Hebrew UI is right-to-left', async ({ page }) => {
  await page.goto('./?sim=1&lang=he');
  await expect(page.getByTestId('btn-start')).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.dir)).toBe('rtl');
  expect(await page.evaluate(() => document.documentElement.lang)).toBe('he');
});
