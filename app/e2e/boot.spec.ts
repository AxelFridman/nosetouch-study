/**
 * Start-up failure (storage blocked: Safari "Block All Cookies", a WebView without IndexedDB): the
 * message is in the participant's language, and the browser's own (English) message stays
 * left-to-right inside it, so a Hebrew sentence keeps its punctuation in place.
 */
import { expect, test } from './fixtures';
import { en } from '../src/i18n/en';
import { he } from '../src/i18n/he';

const BROWSER_MESSAGE = 'The operation is insecure.';

for (const [lang, strings, dir] of [
  ['en', en, 'ltr'],
  ['he', he, 'rtl'],
] as const) {
  test(`boot failure is shown in ${lang}`, async ({ page }) => {
    await page.addInitScript((message) => {
      Object.defineProperty(IDBFactory.prototype, 'open', {
        configurable: true,
        value: () => {
          throw new DOMException(message, 'SecurityError');
        },
      });
    }, BROWSER_MESSAGE);
    await page.goto(`./?sim=1&lang=${lang}`);
    const screen = page.getByTestId('screen-boot-failed');
    await expect(screen).toBeVisible();
    await expect(screen.locator('h1')).toHaveText(strings.common.appName);
    const detail = screen.locator('bdi[dir="ltr"]');
    await expect(detail).toHaveText(BROWSER_MESSAGE);
    const [before] = strings.common.bootFailed('\u0000').split('\u0000');
    await expect(screen.getByRole('alert')).toContainText(before.trim());
    expect(await page.evaluate(() => document.documentElement.dir)).toBe(dir);
  });
}
