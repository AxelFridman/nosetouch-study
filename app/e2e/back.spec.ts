/**
 * Back and pause (spec v2 §1, §3). The system back button (Android back, iOS edge swipe =
 * history.back) never leaves the app in the middle of a session: it opens the pause sheet, also
 * during a recording (the trial stops; an attempt that never reached GO is discarded) and however
 * often it is pressed. Outside the session it goes where the screen's ← arrow goes (profile →
 * welcome, done → welcome). The live panel's pause control needs an ≈800 ms hold: a nose touch
 * (80–240 ms) must never trigger it. "Redo last movement" / "Last one was the wrong hand" revoke
 * the last accepted trial and run it again; "Continue later" keeps the session resumable.
 * Simulation mode; data-testid selectors only.
 */
import type { Page } from '@playwright/test';
import { expect, silenceLog, test } from './fixtures';
import {
  currentView,
  disableWebShare,
  expectAbove,
  exportBundle,
  fillProfile,
  finishFromPause,
  onboard,
  openApp,
  setToggle,
  tapFresh,
  watchErrors,
} from './helpers';
import { en } from '../src/i18n/en';

/** Waits for a trial on the live panel whose index satisfies `pred`, in its hold phase (before GO). */
async function waitForHold(page: Page, pred: (trial: number) => boolean = () => true, timeout = 45_000): Promise<number> {
  let found: number | null = null;
  await expect
    .poll(
      async () => {
        const v = await currentView(page);
        found = v?.id === 'trial-panel' && v.phase === 'waiting_still' && v.trial !== null && pred(v.trial) ? v.trial : null;
        return found;
      },
      { timeout, intervals: [20] },
    )
    .not.toBeNull();
  return found as unknown as number;
}

/**
 * The next trial (index > `after`) in its hold phase, as early as possible; a block intro on the way
 * is started. Re-runs of the same index (a failed simulated trial) are skipped.
 */
async function nextHold(page: Page, after: number, timeout = 60_000): Promise<number> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const v = await currentView(page);
    if (v?.id === 'block-intro') {
      await tapFresh(page, page.getByTestId('btn-block-start'));
      continue;
    }
    if (v?.id === 'trial-panel' && v.phase === 'waiting_still' && v.trial !== null && v.trial > after) return v.trial;
    await page.waitForTimeout(20);
  }
  throw new Error(`no trial after ${after} reached its hold phase`);
}

/** The trial index on the live panel (null when another screen is up). */
async function panelTrial(page: Page): Promise<number | null> {
  const v = await currentView(page);
  return v?.id === 'trial-panel' ? v.trial : null;
}

/** Presses the live panel's pause control for `ms` (pointer down … up). */
async function pressPause(page: Page, ms: number): Promise<void> {
  await page.getByTestId('btn-pause').click({ delay: ms, timeout: 5_000 });
}

test('back never leaves a session: the pause sheet from every step; redo and wrong hand re-run the last trial', async ({ page }) => {
  test.setTimeout(6 * 60_000);
  const errors = watchErrors(page);
  await disableWebShare(page);
  await openApp(page);
  const appUrl = page.url();

  // outside a session, back goes where the ← arrow goes: profile → welcome (consent kept)
  await setToggle(page.getByTestId('chk-consent'), true);
  await tapFresh(page, page.getByTestId('btn-start'));
  await expect(page.getByTestId('screen-profile')).toBeVisible();
  await page.goBack();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  await onboard(page);
  await expect(page.getByTestId('howto-card')).toBeVisible();

  // How-to card: repeated presses keep the sheet (the guard entry is restored every time)
  const sheet = page.getByTestId('session-exit');
  for (let i = 0; i < 3; i++) {
    await page.goBack();
    await expect(sheet).toBeVisible();
    expect(page.url()).toBe(appUrl);
  }
  await expect(page.getByTestId('pause-reason')).toHaveAttribute('data-reason', 'paused');
  // nothing accepted yet: nothing to redo
  await expect(page.getByTestId('btn-redo-last')).toBeDisabled();
  await expect(page.getByTestId('btn-wrong-hand')).toBeDisabled();
  await tapFresh(page, page.getByTestId('btn-continue-session'));
  await expect(page.getByTestId('howto-card')).toBeVisible();

  // during a recording: the trial stops, the sheet opens; Continue runs the same trial again at once
  await tapFresh(page, page.getByTestId('btn-try'));
  await expect(page.getByTestId('trial-panel')).toBeVisible();
  await page.goBack();
  await expect(sheet).toBeVisible({ timeout: 10_000 });
  await tapFresh(page, page.getByTestId('btn-continue-session'));
  await expect(page.getByTestId('trial-panel')).toHaveAttribute('data-trial-index', '0');
  await tapFresh(page, page.getByTestId('btn-hand-right'), 45_000);

  // first block: once its first trial is accepted, back → Redo last movement
  const intro = page.getByTestId('block-intro');
  await expect(intro).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-block-start'));
  const first = await waitForHold(page);
  await waitForHold(page, (i) => i > first); // the chain went on: trial `first` was accepted
  const running = (await panelTrial(page))!;
  await page.goBack();
  await expect(sheet).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId('btn-redo-last')).toBeEnabled();
  await tapFresh(page, page.getByTestId('btn-redo-last'));
  await expect(page.getByTestId('intro-redo')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-block-start'));
  // the revoked trial runs again (it was the one just before the stopped one)
  expect(await waitForHold(page)).toBe(running - 1);

  // after it is accepted again: back → Last one was the wrong hand → the same again
  await waitForHold(page, (i) => i > running - 1);
  const stopped = (await panelTrial(page))!;
  await page.goBack();
  await expect(sheet).toBeVisible({ timeout: 10_000 });
  await tapFresh(page, page.getByTestId('btn-wrong-hand'));
  await expect(page.getByTestId('intro-redo')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-block-start'));
  expect(await waitForHold(page)).toBe(stopped - 1);
  await waitForHold(page, (i) => i > stopped - 1);

  // finish from the sheet; on the done screen back goes to the start screen, like its Home button
  await page.goBack();
  await expect(sheet).toBeVisible({ timeout: 10_000 });
  await finishFromPause(page);
  await expect(page.getByTestId('screen-done')).toBeVisible({ timeout: 30_000 });
  const bundle = await exportBundle(page);
  await page.goBack();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();

  // the two revoked trials are kept, not accepted, and were accepted again in a new attempt
  const revoked = bundle.trials.filter((t) => t.outcome.note === 'redo_requested');
  expect(revoked.map((t) => [t.trial_index, t.outcome.participant_flag])).toEqual(
    expect.arrayContaining([
      [running - 1, null],
      [stopped - 1, 'wrong_hand'],
    ]),
  );
  for (const r of revoked) {
    expect(r.outcome.accepted).toBe(false);
    const again = bundle.trials.filter((t) => t.trial_index === r.trial_index && t.outcome.accepted);
    expect(again, `trial ${r.trial_index} accepted again`).toHaveLength(1);
    expect(again[0].attempt).toBeGreaterThan(r.attempt);
  }
  // a stopped attempt that never reached GO was discarded; every kept attempt has its GO
  expect(bundle.trials.every((t) => t.timing.go_ms !== null)).toBe(true);
  expect(bundle.sessions[0].status).toBe('ended_early');
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
  expect(errors.consoleErrors, errors.consoleErrors.join('\n')).toEqual([]);
});

test('the pause control: a short press does nothing, a long hold pauses; Continue later keeps the session resumable', async ({ page }) => {
  test.setTimeout(5 * 60_000);
  const errors = watchErrors(page);
  await openApp(page);
  await onboard(page);
  await tapFresh(page, page.getByTestId('btn-try'));

  // a 200 ms press (a nose touch is 80–240 ms) during the hold: nothing happens, the trial goes on
  const free = await waitForHold(page);
  await pressPause(page, 200);
  await expect
    .poll(async () => (await currentView(page))?.phase ?? null, { timeout: 15_000 })
    .toMatch(/armed|moving|post_roll|returning/);
  await expect(page.getByTestId('session-exit')).toHaveCount(0);
  expect(free).toBe(0);
  await tapFresh(page, page.getByTestId('btn-hand-right'), 45_000);
  await tapFresh(page, page.getByTestId('btn-block-start'));

  // an 800 ms hold before the movement pauses (the simulated hold is short: retry on the next trial)
  let paused: number | null = null;
  let after = -1;
  for (let attempt = 0; attempt < 8 && paused === null; attempt++) {
    const trial = await nextHold(page, after);
    after = trial;
    await pressPause(page, 1000).catch(() => undefined); // the control turns disabled at the movement
    const sheetOpen = await expect(page.getByTestId('session-exit'))
      .toBeVisible({ timeout: 1_000 })
      .then(() => true)
      .catch(() => false);
    if (sheetOpen) {
      paused = trial;
      console.log(`long press: paused trial ${trial} on press ${attempt + 1}`);
    }
  }
  expect(paused, 'a long press paused one of the trials before its movement').not.toBeNull();
  await expect(page.getByTestId('pause-reason')).toHaveAttribute('data-reason', 'paused');

  // Continue later → the start screen offers to resume; resuming opens the current block's intro
  await tapFresh(page, page.getByTestId('btn-exit-pause'));
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  await expect(page.getByTestId('resume-card')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-resume-session'));
  await expect(page.getByTestId('block-intro')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-block-start'));
  // the paused trial (never reached GO, so not saved) is the next one to run
  expect(await waitForHold(page)).toBe(paused);
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
  expect(errors.consoleErrors, errors.consoleErrors.join('\n')).toEqual([]);
});

test('a returning participant resumes after a reload', async ({ page }) => {
  const errors = watchErrors(page);
  await openApp(page);
  await setToggle(page.getByTestId('chk-consent'), true);
  await tapFresh(page, page.getByTestId('btn-start'));
  await fillProfile(page);
  await tapFresh(page, page.getByTestId('btn-save-profile'));
  await expect(page.getByTestId('howto-card')).toBeVisible({ timeout: 20_000 });
  // the page is reloaded mid-session (a phone call, the browser killed the tab...)
  await page.reload();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  await expect(page.getByTestId('returning-card')).toBeVisible();
  await expect(page.getByTestId('resume-card')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-resume-session'));
  await expect(page.getByTestId('howto-card')).toBeVisible();
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
});

test('after a reload, Resume comes first on a phone-sized screen and the visit goes on where it stopped', async ({ page }) => {
  test.setTimeout(3 * 60_000);
  const errors = watchErrors(page);
  await page.setViewportSize({ width: 390, height: 664 }); // the visible viewport of an iPhone 12–15 in Safari
  await openApp(page);
  await onboard(page);
  // the free trial and the hand question: the first block's intro
  await tapFresh(page, page.getByTestId('btn-try'));
  await tapFresh(page, page.getByTestId('btn-hand-right'), 45_000);
  await expect(page.getByTestId('block-intro')).toHaveAttribute('data-block-index', '1');

  // the phone reloads the page (an app switch, a call, a discarded tab, the link opened again)
  await page.reload();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  const resume = page.getByTestId('btn-resume-session');
  await expect(resume).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  // above the sticky footer (consent + Start), without scrolling
  await expectAbove(page, resume, page.getByTestId('chk-consent'), 'welcome after a reload: Resume');
  await expect(page.getByTestId('btn-start')).toHaveText(en.onboarding.welcome.startOver);

  await tapFresh(page, resume);
  // block 1's intro again, not the How-to card; on a fresh page the hand is named plainly
  const intro = page.getByTestId('block-intro');
  await expect(intro).toHaveAttribute('data-block-index', '1');
  await expect(intro).toHaveAttribute('data-intro-line', 'phoneIn');
  await expect(intro).toHaveAttribute('data-switch', '0');
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
});
