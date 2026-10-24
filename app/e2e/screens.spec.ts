/**
 * Visual QA: walks a whole simulated street visit and saves a screenshot of every screen and card
 * (welcome + consent, profile, How-to card, live panel phases, hand question, every block intro, pause
 * sheet, checkpoint, done, and the menu screens). Opt-in (not part of the normal e2e run):
 *   SCREENS=1 SCREENS_LANGS=en,he SCREENS_DIR=<folder> SCREENS_VIEWPORT=360x640 npx playwright test e2e/screens.spec.ts
 * Defaults: languages "en", folder test-results/screens, the Pixel 7 viewport of the project.
 * Screenshots are full-page; e2e/layout.spec.ts checks what is visible without scrolling.
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import type { Page } from '@playwright/test';
import { expect, test } from './fixtures';
import { disableWebShare, fillProfile, openApp, runSession, setToggle, tapFresh, watchErrors } from './helpers';

const LANGS = (process.env.SCREENS_LANGS ?? 'en').split(',').map((s) => s.trim()).filter(Boolean) as ('en' | 'he')[];
const DIR = process.env.SCREENS_DIR ?? 'test-results/screens';
const VIEWPORT = /^(\d+)x(\d+)$/.exec(process.env.SCREENS_VIEWPORT ?? '');
if (VIEWPORT) test.use({ viewport: { width: Number(VIEWPORT[1]), height: Number(VIEWPORT[2]) } });

for (const lang of LANGS) {
  test(`screenshots (${lang})`, async ({ page }) => {
    test.skip(!process.env.SCREENS, 'set SCREENS=1 to capture screenshots');
    test.setTimeout(12 * 60_000);
    const out = join(DIR, lang);
    await mkdir(out, { recursive: true });
    let n = 0;
    const snap = async (name: string, fullPage = true): Promise<void> => {
      n += 1;
      await page.screenshot({ path: join(out, `${String(n).padStart(2, '0')}-${name}.png`), fullPage });
    };
    const errors = watchErrors(page);
    await disableWebShare(page);

    await openApp(page, { lang });
    await snap('welcome');
    await page.getByTestId('consent-details').locator('summary').click();
    await snap('welcome-full-information');
    await setToggle(page.getByTestId('chk-consent'), true);
    await tapFresh(page, page.getByTestId('btn-start'));

    await expect(page.getByTestId('screen-profile')).toBeVisible();
    await snap('profile-empty');
    await fillProfile(page);
    await snap('profile-filled');
    await tapFresh(page, page.getByTestId('btn-save-profile'));

    await expect(page.getByTestId('howto-card')).toBeVisible({ timeout: 20_000 });
    await snap('howto-card');
    await tapFresh(page, page.getByTestId('btn-try'));
    await expect(page.getByTestId('trial-panel')).toBeVisible();
    await snap('live-free-trial', false);
    await expect(page.getByTestId('hand-report')).toBeVisible({ timeout: 45_000 });
    await snap('hand-question');
    await tapFresh(page, page.getByTestId('btn-hand-right'));

    const phases = new Set<string>();
    let pauseShot = false;
    await runSession(page, {
      onIntro: async ({ block, page: p }) => {
        await snap(`block-intro-${block}`);
        if (!pauseShot && (await p.getByTestId('intro-encourage').isVisible())) {
          pauseShot = true;
          await tapFresh(p, p.getByTestId('btn-session-exit'));
          await expect(p.getByTestId('pause-sheet')).toBeVisible();
          await snap('pause-sheet');
          await tapFresh(p, p.getByTestId('btn-continue-session'));
          await expect(p.getByTestId('block-intro')).toBeVisible();
        }
        return 'start' as const;
      },
      onTrial: async ({ phase, page: p }) => {
        if (phase && !phases.has(phase)) {
          phases.add(phase);
          await snap(`live-${phase}`, false);
        } else {
          await p.waitForTimeout(80);
        }
      },
      onCheckpoint: async () => {
        await snap('checkpoint');
        return 'finish' as const;
      },
      deadlineMs: 10 * 60_000,
    });
    await expect(page.getByTestId('screen-done')).toBeVisible();
    await snap('done');

    await page.getByTestId('btn-home').click();
    await expect(page.getByTestId('btn-start')).toBeVisible();
    await snap('welcome-returning');
    for (const [btn, name] of [
      ['btn-menu-data', 'data'],
      ['btn-menu-settings', 'settings'],
      ['btn-menu-about', 'about'],
      ['btn-menu-sandbox', 'sandbox'],
    ] as const) {
      await page.getByTestId(btn).click();
      await page.waitForTimeout(400);
      if (name === 'sandbox') {
        const start = page.getByTestId('btn-sandbox-start');
        if (await start.isVisible()) {
          await start.click();
          await page.waitForTimeout(6000); // let the simulator produce a movement or two
        }
      }
      await snap(name);
      await goHome(page);
    }
    expect(errors.pageErrors, `uncaught errors in the page:\n${errors.pageErrors.join('\n')}`).toEqual([]);
  });
}

async function goHome(page: Page): Promise<void> {
  const back = page.getByTestId('btn-back');
  if (await back.isVisible().catch(() => false)) await back.click();
  else await page.goto(page.url());
  await expect(page.getByTestId('btn-start')).toBeVisible({ timeout: 15_000 });
}
