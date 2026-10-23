/**
 * Small phone viewports: what the participant must follow is on screen above the sticky footer
 * button, without scrolling. 360×640 is a small Android phone, 390×664 the visible viewport of an
 * iPhone 12–15 in Safari. The trial has the longest instruction of the main blocks (sideways and
 * facing away × fast × in line with the shoulder). Screenshots go to LAYOUT_DIR (default
 * test-results/layout). data-testid selectors only; texts come from the string tables.
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { expect, test, type Locator } from '@playwright/test';
import { en } from '../src/i18n/en';
import { he } from '../src/i18n/he';
import { chooseCustomProtocol, onboardToSetup, tapFresh } from './helpers';

const DIR = process.env.LAYOUT_DIR ?? 'test-results/layout';
const VIEWPORTS = [
  { width: 360, height: 640 },
  { width: 390, height: 664 },
];
const STRINGS = { en, he };

async function box(l: Locator) {
  const b = await l.boundingBox();
  if (!b) throw new Error('element is not rendered');
  return b;
}

/** `item` lies inside the viewport, entirely above the top edge of `button`. */
async function expectAbove(item: Locator, button: Locator, what: string): Promise<void> {
  const a = await box(item);
  const b = await box(button);
  expect(a.y, `${what}: top edge`).toBeGreaterThanOrEqual(0);
  expect(a.y + a.height, `${what}: bottom edge vs the button's top`).toBeLessThanOrEqual(b.y + 0.5);
}

for (const vp of VIEWPORTS) {
  test.describe(`${vp.width}×${vp.height}`, () => {
    test.use({ viewport: vp });

    for (const lang of ['en', 'he'] as const) {
      test(`instructions stay above the sticky button (${lang})`, async ({ page }) => {
        const out = join(DIR, `${lang}-${vp.width}x${vp.height}`);
        await mkdir(out, { recursive: true });
        const shot = (name: string) => page.screenshot({ path: join(out, `${name}.png`) });

        await onboardToSetup(page, lang);
        await chooseCustomProtocol(page, { orientations: ['top_right_away'], speeds: ['fast'], starts: ['shoulder'] });
        await page.evaluate(() => window.scrollTo(0, 0));
        const start = page.getByTestId('btn-start-session');
        await expectAbove(page.getByTestId('safety-note'), start, 'setup safety note');
        await shot('1-setup');

        await tapFresh(page, start);
        const blockStart = page.getByTestId('btn-block-start');
        await expect(blockStart).toBeVisible();
        await expectAbove(page.getByTestId('intro-safety'), blockStart, 'block intro safety notice');
        await shot('2-intro');

        await tapFresh(page, blockStart);
        const ready = page.getByTestId('btn-ready');
        await expect(ready).toBeEnabled();
        for (const row of ['instr-phone', 'instr-speed', 'instr-start', 'instr-movement']) {
          await expectAbove(page.getByTestId(row), ready, row);
        }
        await expect(page.getByTestId('ready-figure')).toBeHidden(); // short viewport: no figure
        await shot('3-ready');

        // the live panel names the start position next to the speed
        await tapFresh(page, ready);
        const s = STRINGS[lang].common;
        await expect(page.getByTestId('tp-condition')).toContainText(s.startPositions.shoulder.label);
        await expect(page.getByTestId('tp-condition')).toContainText(s.speeds.fast.label);
        await shot('4-recording');
      });
    }
  });
}
