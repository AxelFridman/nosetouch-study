/**
 * Double taps and stray touches. Consecutive steps put their main button in the same sticky-footer
 * slot, and .btn has touch-action: manipulation, so each tap of a double tap is its own click: the
 * second one must not press the NEXT screen's button (the app ignores taps for TAP_GUARD_MS after a
 * step change). Screens that appear without a tap (the hand question after the free trial: the phone
 * may still be at the face) keep their buttons disabled for the arm delay, so a cheek or a nose cannot
 * answer. And on the live panel a short touch of the pause control (a nose touch) does nothing.
 * Simulation mode; data-testid selectors only.
 */
import { expect, test } from './fixtures';
import { currentView, fillProfile, openApp, setToggle, TAP_DELAY_MS, tapFresh, watchErrors } from './helpers';
import { en } from '../src/i18n/en';

test('the second tap of a double tap never presses the next step’s button', async ({ page }) => {
  const errors = watchErrors(page);
  await openApp(page);

  // welcome Start → the profile's Continue sits under the finger: no empty-form validation
  await setToggle(page.getByTestId('chk-consent'), true);
  await page.waitForTimeout(TAP_DELAY_MS);
  await page.getByTestId('btn-start').dblclick();
  await expect(page.getByTestId('screen-profile')).toBeVisible();
  await page.waitForTimeout(TAP_DELAY_MS);
  await expect(page.getByTestId('screen-profile')).toBeVisible();
  await expect(page.getByText(en.onboarding.profile.errors.fixBelow)).toHaveCount(0);

  // profile Continue → "Try it once" sits under the finger: the free trial must not start by itself
  await fillProfile(page);
  await page.getByTestId('btn-save-profile').dblclick();
  await expect(page.getByTestId('howto-card')).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(TAP_DELAY_MS);
  await expect(page.getByTestId('howto-card')).toBeVisible();
  await expect(page.getByTestId('trial-panel')).toHaveCount(0);

  // "Try it once" → the live panel's pause control sits under the finger: one trial, never a pause
  await tapFresh(page, page.getByTestId('btn-try'));
  await expect(page.getByTestId('trial-panel')).toBeVisible();
  await page.getByTestId('trial-panel').getByTestId('btn-pause').click({ delay: 120 }); // a nose-length touch
  await page.waitForTimeout(1_000);
  await expect(page.getByTestId('session-exit')).toHaveCount(0);

  // the hand question appears without a tap: a touch the moment it appears is ignored (arm delay)
  const hand = page.getByTestId('hand-report');
  await expect(hand).toBeVisible({ timeout: 45_000 });
  await page.getByTestId('btn-hand-left').click({ force: true, timeout: 2_000 }).catch(() => undefined);
  await page.waitForTimeout(300);
  await expect(hand).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-hand-right'));

  // block Start → one trial starts; the second tap lands on the live panel and changes nothing
  const blockStart = page.getByTestId('btn-block-start');
  await expect(blockStart).toBeVisible();
  await expect(blockStart).toBeEnabled();
  await page.waitForTimeout(TAP_DELAY_MS);
  await blockStart.dblclick();
  await expect(page.getByTestId('trial-panel')).toBeVisible();
  await page.waitForTimeout(1_000);
  await expect(page.getByTestId('session-exit')).toHaveCount(0);
  expect((await currentView(page))?.id).toBe('trial-panel');
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
});
