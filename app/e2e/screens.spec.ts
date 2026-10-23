/**
 * Visual QA: walks the simulated journey and saves a screenshot of every screen.
 * Opt-in (not part of the normal e2e run):
 *   SCREENS=1 SCREENS_LANGS=en,he SCREENS_DIR=<folder> SCREENS_VIEWPORT=360x640 npx playwright test e2e/screens.spec.ts
 * Defaults: languages "en", folder test-results/screens, the Pixel 7 viewport of the project.
 * Screenshots are full-page; e2e/layout.spec.ts checks what is visible above the sticky buttons.
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';
import { chooseTwoTrialProtocol, disableWebShare, fillProfile, runCalibrationStep, runTrial, tapFresh } from './helpers';

const LANGS = (process.env.SCREENS_LANGS ?? 'en').split(',').map((s) => s.trim()).filter(Boolean);
const DIR = process.env.SCREENS_DIR ?? 'test-results/screens';
const VIEWPORT = /^(\d+)x(\d+)$/.exec(process.env.SCREENS_VIEWPORT ?? '');
if (VIEWPORT) test.use({ viewport: { width: Number(VIEWPORT[1]), height: Number(VIEWPORT[2]) } });

for (const lang of LANGS) {
  test(`screenshots (${lang})`, async ({ page }) => {
    test.skip(!process.env.SCREENS, 'set SCREENS=1 to capture screenshots');
    test.setTimeout(300_000);
    const out = join(DIR, lang);
    await mkdir(out, { recursive: true });
    let n = 0;
    const snap = async (name: string): Promise<void> => {
      n += 1;
      await page.screenshot({ path: join(out, `${String(n).padStart(2, '0')}-${name}.png`), fullPage: true });
    };
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await disableWebShare(page);

    await page.goto(`/?sim=1&lang=${lang}`);
    await expect(page.getByTestId('btn-start')).toBeVisible();
    await snap('welcome');
    await page.getByTestId('btn-start').click();
    await expect(page.getByTestId('chk-consent')).toBeVisible();
    await snap('consent');
    await page.getByTestId('chk-consent').click();
    await page.getByTestId('btn-consent-accept').click();

    await expect(page.getByTestId('in-age')).toBeVisible();
    await snap('profile-empty');
    await fillProfile(page);
    await snap('profile-filled');
    await page.getByTestId('btn-save-profile').click();

    await expect(page.getByTestId('btn-request-permission')).toBeVisible();
    await snap('sensors-before');
    await page.getByTestId('btn-request-permission').click();
    await expect(page.getByTestId('btn-continue')).toBeEnabled({ timeout: 30_000 });
    await snap('sensors-after');
    await page.getByTestId('btn-continue').click();

    await expect(page.getByTestId('btn-calibrate-flat')).toBeVisible();
    await snap('calibration-start');
    await runCalibrationStep(page, 'btn-calibrate-flat');
    await runCalibrationStep(page, 'btn-calibrate-upright');
    await snap('calibration-done');
    await page.getByTestId('btn-continue').click();

    await expect(page.getByTestId('opt-preset-standard')).toBeVisible();
    await snap('setup-presets');
    await chooseTwoTrialProtocol(page);
    await snap('setup-custom');
    await tapFresh(page, page.getByTestId('btn-start-session'));

    const nextBlock = page.getByTestId('btn-next-block');
    const doneScreen = page.getByTestId('screen-done');
    for (let trial = 0; trial < 2; trial++) {
      await runTrial(page, trial === 0 ? snap : async () => undefined);
      await expect(
        nextBlock.or(page.getByTestId('btn-block-start')).or(page.getByTestId('btn-ready')).or(doneScreen).first(),
      ).toBeVisible({ timeout: 30_000 });
      if (await nextBlock.isVisible()) {
        await snap('block-break');
        await tapFresh(page, nextBlock);
      }
    }
    await expect(doneScreen).toBeVisible({ timeout: 30_000 });
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
    expect(errors, `uncaught errors in the page:\n${errors.join('\n')}`).toEqual([]);
  });
}

async function goHome(page: Page): Promise<void> {
  const back = page.getByTestId('btn-back');
  if (await back.isVisible().catch(() => false)) await back.click();
  else await page.goto(page.url());
  await expect(page.getByTestId('btn-start')).toBeVisible({ timeout: 15_000 });
}
