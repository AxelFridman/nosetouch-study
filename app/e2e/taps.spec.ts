/**
 * Double taps: consecutive steps put their main button in the same footer slot, and .btn has
 * touch-action: manipulation, so each tap of a double tap is its own click. The second one must not
 * press the next step's button (start a session from the calibration screen, start a recording
 * from the block intro). Simulation mode; data-testid selectors only.
 */
import { expect, test } from '@playwright/test';
import { TAP_GUARD_MS } from '../src/app/state';
import { chooseTwoTrialProtocol, fillProfile, inputOf, tapFresh } from './helpers';

test('the second tap of a double tap never presses the next step’s button', async ({ page }) => {
  await page.goto('/?sim=1&lang=en');
  await page.getByTestId('btn-start').click();
  await page.getByTestId('chk-consent').click();
  await expect(await inputOf(page.getByTestId('chk-consent'))).toBeChecked();
  await page.getByTestId('btn-consent-accept').click();
  await fillProfile(page);
  await page.getByTestId('btn-save-profile').click();
  await page.getByTestId('btn-request-permission').click();
  await expect(page.getByTestId('btn-continue')).toBeEnabled({ timeout: 30_000 });
  await page.getByTestId('btn-continue').click();

  // calibration → setup: "Start recording" appears under the finger
  const skip = page.getByTestId('btn-skip-calibration');
  await expect(skip).toBeVisible();
  await page.waitForTimeout(TAP_GUARD_MS + 150);
  await skip.dblclick();
  await expect(page.getByTestId('screen-setup')).toBeVisible();
  await page.waitForTimeout(TAP_GUARD_MS + 150);
  await expect(page.getByTestId('screen-setup')).toBeVisible(); // no session was started

  // block intro → Ready: the recording must not start before the instruction was on screen
  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));
  const blockStart = page.getByTestId('btn-block-start');
  await expect(blockStart).toBeVisible();
  await page.waitForTimeout(TAP_GUARD_MS + 150);
  await blockStart.dblclick();
  await expect(page.getByTestId('trial-ready')).toBeVisible();
  await page.waitForTimeout(TAP_GUARD_MS + 150);
  await expect(page.getByTestId('trial-panel')).toHaveCount(0);
  await expect(page.getByTestId('btn-ready')).toBeEnabled();
});
