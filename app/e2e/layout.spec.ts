/**
 * Small phone viewports, English and Hebrew: the v2 cards fit 360×640 (a small Android phone) and
 * 390×664 (the visible viewport of an iPhone 12–15 in Safari). What the participant must see or tap
 * is on screen without scrolling — above the sticky footer button where there is one — and no page
 * ever scrolls sideways. The live panel (read at arm's length, no free hand to scroll) must not
 * scroll at all. The checkpoint card is checked at both sizes by the full journeys (flow.spec.ts in
 * English, rtl.spec.ts in Hebrew). Screenshots go to LAYOUT_DIR (default test-results/layout).
 * Sideways grips may rotate the page, so a sideways block also runs in landscape. data-testid selectors
 * only; the only grip tile used is top_left_toward (never *_away).
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { Page } from '@playwright/test';
import { expect, test } from './fixtures';
import {
  chooseCustomProtocol,
  currentView,
  expectAbove,
  expectInViewport,
  expectNoHorizontalScroll,
  expectNoVerticalScroll,
  fillProfile,
  finishFromPause,
  onboard,
  openApp,
  runSession,
  setToggle,
  SMALL_VIEWPORTS,
  tapFresh,
  watchErrors,
} from './helpers';

const DIR = process.env.LAYOUT_DIR ?? 'test-results/layout';

/** The live panel: hand and speed words, the status line and the pause control, no scrolling. */
async function checkLivePanel(page: Page, what: string): Promise<void> {
  await expect(page.getByTestId('trial-panel')).toBeVisible();
  for (const id of ['tp-hand', 'tp-speed', 'tp-status', 'btn-pause']) {
    await expectInViewport(page, page.getByTestId(id), `${what}: ${id}`);
  }
  await expectNoVerticalScroll(page, what);
  await expectNoHorizontalScroll(page, what);
}

/**
 * Sideways grips (street tier 2) rotate the page when the phone's auto-rotation is on (iOS web pages
 * cannot lock it): the block intro of a sideways block and its live panel must work in landscape too.
 * 640×360: a small Android phone turned; 844×350: an iPhone 12–15 turned, Safari's bars included.
 */
const LANDSCAPE = [
  { width: 640, height: 360 },
  { width: 844, height: 350 },
] as const;

for (const lang of ['en', 'he'] as const) {
  test(`sideways blocks in landscape: block intro and live panel (${lang})`, async ({ page }) => {
    test.setTimeout(4 * 60_000);
    const errors = watchErrors(page);
    const out = join(DIR, `${lang}-landscape`);
    await mkdir(out, { recursive: true });
    await openApp(page, { lang, setup: true });
    await onboard(page, { until: 'setup' });
    // one sideways trial per hand (researchers' custom plan)
    await chooseCustomProtocol(page, { orientations: ['top_left_toward'], speeds: ['natural'], starts: ['natural'] });
    await tapFresh(page, page.getByTestId('btn-start-session'));
    for (const [i, vp] of LANDSCAPE.entries()) {
      const start = page.getByTestId('btn-block-start');
      await expect(start).toBeVisible({ timeout: 45_000 });
      await page.setViewportSize(vp); // the participant turns the phone sideways while reading the intro
      await page.evaluate(() => window.scrollTo(0, 0));
      const what = `${vp.width}×${vp.height}`;
      await expectInViewport(page, start, `block intro ${what}: Start`);
      await expectInViewport(page, page.getByTestId('intro-hand'), `block intro ${what}: hand`);
      await expectNoHorizontalScroll(page, `block intro ${what}`);
      await page.screenshot({ path: join(out, `${i + 1}a-intro-${vp.width}x${vp.height}.png`) });
      await tapFresh(page, start);
      await checkLivePanel(page, `live panel ${what}`);
      await page.screenshot({ path: join(out, `${i + 1}b-live-${vp.width}x${vp.height}.png`) });
    }
    await expect(page.getByTestId('screen-done')).toBeVisible({ timeout: 45_000 });
    await expectInViewport(page, page.getByTestId('btn-next-person'), 'done (landscape): Next person');
    expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
  });
}

for (const vp of SMALL_VIEWPORTS) {
  test.describe(`${vp.width}×${vp.height}`, () => {
    test.use({ viewport: vp });

    for (const lang of ['en', 'he'] as const) {
      test(`the v2 cards fit without scrolling (${lang})`, async ({ page }) => {
        test.setTimeout(4 * 60_000);
        const errors = watchErrors(page);
        const out = join(DIR, `${lang}-${vp.width}x${vp.height}`);
        await mkdir(out, { recursive: true });
        const shot = (name: string) => page.screenshot({ path: join(out, `${name}.png`) });

        // welcome + consent: the checkbox and Start in the sticky footer
        await openApp(page, { lang });
        await expectInViewport(page, page.getByTestId('btn-start'), 'welcome: Start');
        await expectInViewport(page, page.getByTestId('chk-consent'), 'welcome: consent checkbox');
        await expectNoHorizontalScroll(page, 'welcome');
        await shot('1-welcome');
        await setToggle(page.getByTestId('chk-consent'), true);
        await tapFresh(page, page.getByTestId('btn-start'));

        // profile
        await expect(page.getByTestId('screen-profile')).toBeVisible();
        await fillProfile(page);
        await page.evaluate(() => window.scrollTo(0, 0));
        await expectInViewport(page, page.getByTestId('btn-save-profile'), 'profile: Continue');
        await expectInViewport(page, page.getByTestId('in-name'), 'profile: name');
        await expectNoHorizontalScroll(page, 'profile');
        await shot('2-profile');
        await tapFresh(page, page.getByTestId('btn-save-profile'));

        // How-to card: the animation starts on screen, "Try it once" in the footer
        await expect(page.getByTestId('howto-card')).toBeVisible({ timeout: 20_000 });
        const demo = (await page.getByTestId('gesture-demo').boundingBox())!;
        expect(demo.y, 'How-to: the animation starts on screen').toBeLessThan(vp.height * 0.6);
        await expectInViewport(page, page.getByTestId('btn-try'), 'How-to: Try it once');
        await expectNoHorizontalScroll(page, 'How-to card');
        await shot('3-howto');
        await tapFresh(page, page.getByTestId('btn-try'));

        // the free trial's live panel
        await checkLivePanel(page, 'live panel (free trial)');
        await shot('4-live-free');

        // hand question: both hand buttons and "not sure" on screen
        await expect(page.getByTestId('hand-report')).toBeVisible({ timeout: 45_000 });
        for (const id of ['btn-hand-left', 'btn-hand-right', 'btn-hand-unsure']) {
          await expectInViewport(page, page.getByTestId(id), `hand question: ${id}`);
        }
        await expectNoHorizontalScroll(page, 'hand question');
        await shot('5-hand');
        await tapFresh(page, page.getByTestId('btn-hand-right'));

        // block intro: the hand instruction above the Start button that names the hand
        const start = page.getByTestId('btn-block-start');
        await expect(start).toBeVisible();
        await expectInViewport(page, start, 'block intro: Start');
        await expectAbove(page, page.getByTestId('intro-hand'), start, 'block intro: hand');
        await expectNoHorizontalScroll(page, 'block intro');
        await shot('6-intro');
        await tapFresh(page, start);

        // a block trial's live panel
        await expect.poll(async () => (await currentView(page))?.trial ?? null).not.toBeNull();
        await checkLivePanel(page, 'live panel (block trial)');
        await shot('7-live');

        // pause sheet (system back): Continue and Finish now reachable without scrolling
        await page.goBack();
        await expect(page.getByTestId('pause-sheet')).toBeVisible({ timeout: 10_000 });
        for (const id of ['pause-reason', 'btn-continue-session', 'btn-finish-now']) {
          await expectInViewport(page, page.getByTestId(id), `pause sheet: ${id}`);
        }
        await expectNoHorizontalScroll(page, 'pause sheet');
        await shot('8-pause');
        await finishFromPause(page);

        // done: thanks, the count and "Next person"
        await expect(page.getByTestId('screen-done')).toBeVisible({ timeout: 30_000 });
        for (const id of ['done-thanks', 'stat-accepted', 'btn-next-person']) {
          await expectInViewport(page, page.getByTestId(id), `done: ${id}`);
        }
        await expectNoHorizontalScroll(page, 'done');
        await shot('9-done');
        expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
      });
    }
  });
}

test('street block intros: what changes stays above Start on small and landscape screens (blocks 3, 5, 7)', async ({ page }) => {
  test.setTimeout(8 * 60_000);
  const errors = watchErrors(page);
  const out = join(DIR, 'en-street-intros');
  await mkdir(out, { recursive: true });
  await openApp(page, { lang: 'en' });
  await onboard(page);
  // the usual iPhone Safari viewport, and an iPhone turned sideways (the page rotates in tier 2)
  const sizes = [
    { width: 390, height: 664 },
    { width: 844, height: 390 },
  ] as const;
  const original = page.viewportSize();
  const checked: number[] = [];
  const run = await runSession(page, {
    onIntro: async ({ block, page: p }) => {
      if (block !== 3 && block !== 5 && block !== 7) return 'start';
      const intro = p.getByTestId('block-intro');
      // the same hand, another grip: "Turn the phone…" with "Same hand: …", no "Switch hands!"
      await expect(intro).toHaveAttribute('data-intro-line', 'turnTo');
      await expect(intro).toHaveAttribute('data-switch', '0');
      await expect(p.getByTestId('intro-same-hand')).toBeVisible();
      for (const vp of sizes) {
        await p.setViewportSize(vp);
        await p.evaluate(() => window.scrollTo(0, 0));
        const what = `block ${block} intro ${vp.width}×${vp.height}`;
        const start = p.getByTestId('btn-block-start');
        await expectInViewport(p, start, `${what}: Start`);
        await expectAbove(p, p.getByTestId('intro-hand'), start, `${what}: the hand line`);
        await expectAbove(p, p.getByTestId('intro-grip'), start, `${what}: the grip`);
        await expectNoHorizontalScroll(p, what);
        await p.screenshot({ path: join(out, `b${block}-${vp.width}x${vp.height}.png`) });
      }
      if (original) await p.setViewportSize(original);
      checked.push(block);
      return block === 7 ? 'stop' : 'start';
    },
    deadlineMs: 7 * 60_000,
  });
  expect(run.end).toBe('stopped');
  expect(checked).toEqual([3, 5, 7]);
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
});
