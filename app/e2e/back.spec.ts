/**
 * The system back button (Android back, iOS edge swipe = history.back) must never leave the app in
 * the middle of a session: it opens the Pause dialog, also during a recording (the trial is stopped
 * and kept as an aborted attempt) and during the review, however often it is pressed. Outside the
 * session it goes where the screen's ← arrow goes. Simulation mode; data-testid selectors only.
 */
import { expect, test, type Page } from '@playwright/test';
import { chooseTwoTrialProtocol, onboardToSetup, runTrial, tapFresh } from './helpers';

/** Ready → recording → … until a complete trial's review is on screen (simulated trials may fail). */
async function recordUntilReview(page: Page): Promise<void> {
  const review = page.getByTestId('trial-review');
  const retry = page.getByTestId('btn-retry');
  for (let attempt = 1; attempt <= 3; attempt++) {
    await tapFresh(page, page.getByTestId('btn-ready'));
    await expect(review.or(retry).first()).toBeVisible({ timeout: 30_000 });
    if (await review.isVisible()) return;
    await tapFresh(page, retry);
  }
  throw new Error('no complete simulated trial in three attempts');
}

test('back never leaves a session: it opens the Pause dialog, from every step', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await onboardToSetup(page);

  // outside a session, back goes where the ← arrow goes
  await page.goBack();
  await expect(page.getByTestId('screen-calibration')).toBeVisible();
  await page.getByTestId('btn-skip-calibration').click();
  await expect(page.getByTestId('screen-setup')).toBeVisible();

  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));
  await expect(page.getByTestId('block-intro')).toBeVisible();
  const appUrl = page.url();
  const dialog = page.getByTestId('session-exit');

  // block intro: repeated presses keep the dialog (the guard entry is restored every time)
  for (let i = 0; i < 3; i++) {
    await page.goBack();
    await expect(dialog).toBeVisible();
    expect(page.url()).toBe(appUrl);
  }
  await page.getByTestId('btn-exit-cancel').click();
  await expect(page.getByTestId('block-intro')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-block-start'));

  // during a recording: the trial stops like the Stop control, then the dialog opens
  await tapFresh(page, page.getByTestId('btn-ready'));
  await expect(page.getByTestId('trial-panel')).toBeVisible();
  await page.goBack();
  await expect(dialog).toBeVisible({ timeout: 10_000 });
  await page.getByTestId('btn-exit-cancel').click();
  await expect(page.getByTestId('trial-failed')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-retry'));

  // during the review: the dialog, and "Keep going" brings the same review back
  await recordUntilReview(page);
  await page.goBack();
  await expect(dialog).toBeVisible();
  await page.getByTestId('btn-exit-cancel').click();
  await expect(page.getByTestId('trial-review')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-next'));

  // the session then simply goes on: block break → second block → done
  const nextBlock = page.getByTestId('btn-next-block');
  await expect(nextBlock).toBeVisible({ timeout: 30_000 });
  await tapFresh(page, nextBlock);
  await runTrial(page);
  await expect(page.getByTestId('screen-done')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('stat-accepted')).toContainText('2 / 2');
  // the stopped attempt was kept (plus any failed simulated ones): at least 3 attempts for 2 trials
  const attempts = Number((await page.getByTestId('stat-attempts').innerText()).trim().match(/\d+/)?.[0]);
  expect(attempts).toBeGreaterThanOrEqual(3);

  // done: back goes to the start screen, like its Home button
  await page.goBack();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  expect(pageErrors, `uncaught errors in the page:\n${pageErrors.join('\n')}`).toEqual([]);
});

test('Pause from the back button keeps the session resumable', async ({ page }) => {
  await onboardToSetup(page);
  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));
  await expect(page.getByTestId('block-intro')).toBeVisible();
  await page.goBack();
  await tapFresh(page, page.getByTestId('btn-exit-pause'));
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  await expect(page.getByTestId('resume-card')).toBeVisible();
});
