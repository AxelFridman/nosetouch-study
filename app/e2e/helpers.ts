/**
 * Shared Playwright helpers for the simulated participant journey. Only data-testid selectors are
 * used (never UI text), so every helper works in any UI language.
 */
import { expect, type Locator, type Page } from '@playwright/test';
import { TAP_GUARD_MS } from '../src/app/state';
import { ALL_ORIENTATIONS, ODD_VARIANTS } from '../src/core/constants';
import type { Speed, StartPosition } from '../src/core/types';

export const SPEEDS: Speed[] = ['natural', 'fast', 'slow'];
export const STARTS: StartPosition[] = ['natural', 'shoulder', 'center'];

/**
 * Clicks a button of a step that has just appeared. The app ignores taps within TAP_GUARD_MS of a
 * step change (its double-tap guard), and Playwright is faster than any participant.
 */
export async function tapFresh(page: Page, button: Locator): Promise<void> {
  await expect(button).toBeEnabled({ timeout: 30_000 });
  await page.waitForTimeout(TAP_GUARD_MS + 150);
  await button.click();
}

/** The checkbox/radio behind a tile or label: the element itself if it is an input, else its inner input. */
export async function inputOf(el: Locator): Promise<Locator> {
  const tag = await el.evaluate((node) => node.tagName.toLowerCase());
  return tag === 'input' ? el : el.locator('input').first();
}

/** Makes the checked tiles of a MultiChoice group exactly `wanted`, clicking only what must change. */
export async function setTiles(page: Page, prefix: string, all: readonly string[], wanted: readonly string[]): Promise<void> {
  // tick the wanted tiles first so the group never becomes empty (the screen may refuse that)
  for (const pass of ['check', 'uncheck'] as const) {
    for (const value of all) {
      const want = wanted.includes(value);
      if ((pass === 'check') !== want) continue;
      const tile = page.getByTestId(`${prefix}-${value}`);
      if ((await tile.count()) === 0) continue;
      const input = await inputOf(tile);
      if ((await input.isChecked()) !== want) {
        await tile.click();
        await expect(input).toBeChecked({ checked: want });
      }
    }
  }
}

/**
 * Starts one calibration measurement and waits until it has finished: a new "✓" result appeared,
 * or the button is usable again with a different label ("Measure again"), or the step went away.
 */
export async function runCalibrationStep(page: Page, testId: string): Promise<void> {
  const button = page.getByTestId(testId);
  await expect(button).toBeEnabled({ timeout: 30_000 });
  const label = (await button.innerText()).trim();
  const checks = page.getByText('✓');
  const checksBefore = await checks.count();
  await button.click();
  await page.waitForTimeout(300); // let the measurement start (busy state) before polling
  await expect
    .poll(
      async () => {
        if ((await checks.count()) > checksBefore) return true;
        if (!(await button.isVisible())) return true;
        return (await button.isEnabled()) && (await button.innerText()).trim() !== label;
      },
      { timeout: 30_000, intervals: [250] },
    )
    .toBe(true);
}

/** Hook called at interesting moments (used by the screenshot spec). */
export type Snap = (name: string) => Promise<void>;
const noSnap: Snap = async () => undefined;

/** One planned trial: (block intro) → Ready → recording → review → Next. Retries a failed attempt. */
export async function runTrial(page: Page, snap: Snap = noSnap): Promise<void> {
  const blockStart = page.getByTestId('btn-block-start');
  const ready = page.getByTestId('btn-ready');
  const review = page.getByTestId('trial-review');
  const retry = page.getByTestId('btn-retry');
  for (let attempt = 1; attempt <= 3; attempt++) {
    await expect(blockStart.or(ready).first()).toBeVisible({ timeout: 30_000 });
    if (await blockStart.isVisible()) {
      await snap('block-intro');
      await tapFresh(page, blockStart);
    }
    await expect(ready).toBeEnabled({ timeout: 30_000 });
    await snap('trial-ready');
    await tapFresh(page, ready);
    const panel = page.getByTestId('trial-panel');
    await expect(panel).toHaveAttribute('data-phase', /^[a-z_]+$/);
    await snap('trial-recording-wait');
    await expect(panel).toHaveAttribute('data-phase', /moving|post_roll|done/, { timeout: 30_000 }).catch(() => undefined);
    if (await panel.isVisible().catch(() => false)) await snap('trial-recording-moving');
    await expect(review.or(retry).first()).toBeVisible({ timeout: 30_000 });
    if (await review.isVisible()) {
      await snap('trial-review');
      await tapFresh(page, page.getByTestId('btn-next'));
      return;
    }
    console.warn(`simulated trial failed on attempt ${attempt}; trying again`);
    await tapFresh(page, retry);
  }
  throw new Error('a simulated trial failed three times in a row');
}

/** Profile answers for a right-handed simulated participant. */
export async function fillProfile(page: Page): Promise<void> {
  await page.getByTestId('in-age').fill('30');
  for (const id of [
    'opt-sex-female',
    'opt-handed-right',
    'opt-ehi-writing-always_right',
    'opt-ehi-throwing-always_right',
    'opt-ehi-toothbrush-always_right',
    'opt-ehi-spoon-usually_right',
    'opt-phonehand-right',
  ]) {
    await page.getByTestId(id).click();
  }
}

export interface CustomChoice {
  orientations: readonly string[];
  speeds: readonly Speed[];
  starts: readonly StartPosition[];
  odd?: readonly string[];
}

/** Sets up a custom protocol on the setup screen (1 repetition, no free block). */
export async function chooseCustomProtocol(page: Page, choice: CustomChoice): Promise<void> {
  await page.getByTestId('opt-preset-custom').click();
  await expect(page.getByTestId('opt-orient-upright_toward')).toBeVisible();
  await setTiles(page, 'opt-orient', ALL_ORIENTATIONS, choice.orientations);
  await setTiles(page, 'opt-speed', SPEEDS, choice.speeds);
  await setTiles(page, 'opt-start', STARTS, choice.starts);
  await setTiles(page, 'opt-odd', ODD_VARIANTS, choice.odd ?? []);
  await page.getByTestId('in-reps').fill('1');
  const free = page.getByTestId('chk-free');
  if ((await free.count()) > 0 && (await (await inputOf(free)).isChecked())) {
    await free.click();
    await expect(await inputOf(free)).not.toBeChecked();
  }
}

/** Custom protocol: upright_toward × natural speed × natural start × 1 rep, nothing else → 2 trials. */
export async function chooseTwoTrialProtocol(page: Page): Promise<void> {
  await chooseCustomProtocol(page, { orientations: ['upright_toward'], speeds: ['natural'], starts: ['natural'] });
}

/** welcome → consent → profile → sensor check → calibration (skipped) → the setup screen, simulated. */
export async function onboardToSetup(page: Page, lang = 'en'): Promise<void> {
  await page.goto(`./?sim=1&lang=${lang}`);
  await page.getByTestId('btn-start').click();
  const consent = page.getByTestId('chk-consent');
  await consent.click();
  await expect(await inputOf(consent)).toBeChecked();
  await page.getByTestId('btn-consent-accept').click();
  await fillProfile(page);
  await page.getByTestId('btn-save-profile').click();
  await page.getByTestId('btn-request-permission').click();
  const sensorsContinue = page.getByTestId('btn-continue');
  await expect(sensorsContinue).toBeEnabled({ timeout: 30_000 });
  await sensorsContinue.click();
  await page.getByTestId('btn-skip-calibration').click();
  await expect(page.getByTestId('screen-setup')).toBeVisible();
}

/** Forces the download path of deliverExport (no share sheet), whatever the browser supports. */
export async function disableWebShare(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const proto = Navigator.prototype as unknown as Record<string, unknown>;
    for (const key of ['share', 'canShare']) {
      try {
        delete proto[key];
      } catch {
        // not configurable: leave it
      }
    }
  });
}
