/**
 * Hebrew (?lang=he): a whole simulated visit stays right-to-left with Hebrew texts on every card —
 * welcome, How-to card, live panel, hand question (whose LEFT button stays on the physical left),
 * block intros, checkpoint (also on small phones) and done. Texts come from the string tables, never
 * hard-coded; the flow itself uses data-testid selectors only.
 */
import { expect, silenceLog, test } from './fixtures';
import { checkCheckpointLayout, installTimeline, numberIn, onboard, openApp, printTimings, readTimeline, runSession, tapFresh, watchErrors } from './helpers';
import type { Hand } from '../src/core/types';
import { he } from '../src/i18n/he';

/** innerText compared like the screen shows it: whitespace collapsed, ends trimmed. */
const norm = (s: string) => s.replace(/\s+/g, ' ').trim();

test('Hebrew: right-to-left with Hebrew texts on every card of a simulated visit', async ({ page }) => {
  test.setTimeout(12 * 60_000);
  const errors = watchErrors(page);
  await installTimeline(page);
  const dir = () => page.evaluate(() => document.documentElement.dir);

  await openApp(page, { lang: 'he' });
  await expect.poll(dir).toBe('rtl');
  expect(await page.evaluate(() => document.documentElement.lang)).toBe('he');
  await expect(page.getByTestId('screen-welcome').locator('h1')).toHaveText(he.onboarding.welcome.title);
  await expect(page.getByTestId('btn-start')).toHaveText(he.onboarding.welcome.start);

  await onboard(page);
  expect(await dir()).toBe('rtl');
  await expect(page.getByTestId('howto-card')).toContainText(he.session.howTo.steps[0]);
  await expect(page.getByTestId('btn-try')).toHaveText(he.session.howTo.try);

  // the free trial: the live panel names no hand ("your choice")
  await tapFresh(page, page.getByTestId('btn-try'));
  await expect(page.getByTestId('tp-hand')).toHaveText(he.session.live.anyHand);

  // hand question: Hebrew texts, and the LEFT hand's button on the physical left of an RTL page
  await expect(page.getByTestId('hand-report')).toBeVisible({ timeout: 45_000 });
  await expect(page.getByTestId('hand-report')).toContainText(he.session.hand.question);
  await expect(page.getByTestId('btn-hand-left')).toHaveText(he.session.hand.left);
  await expect(page.getByTestId('btn-hand-right')).toHaveText(he.session.hand.right);
  const left = (await page.getByTestId('btn-hand-left').boundingBox())!;
  const right = (await page.getByTestId('btn-hand-right').boundingBox())!;
  expect(left.x + left.width / 2, 'the left-hand button is left of the right-hand one').toBeLessThan(right.x + right.width / 2);
  await tapFresh(page, page.getByTestId('btn-hand-right'));
  await expect(page.getByTestId('block-intro')).toBeVisible();

  const speedWords = new Set(Object.values(he.session.live.speedWords).map(norm));
  let introChecked = false;
  let panelChecked = false;
  let checkpointChecked = false;
  let introHand: Hand | null = null;
  const run = await runSession(page, {
    onIntro: async ({ page: p }) => {
      expect(await dir()).toBe('rtl');
      if (!introChecked) {
        introChecked = true;
        const text = norm(await p.getByTestId('intro-hand').innerText());
        introHand = (['left', 'right'] as Hand[]).find((h) => [he.session.intro.phoneIn(h), he.session.intro.keepIn(h)].map(norm).includes(text)) ?? null;
        expect(introHand, `intro-hand "${text}" is a Hebrew hand instruction`).not.toBeNull();
        await expect(p.getByTestId('btn-block-start')).toHaveText(he.session.intro.startIn(introHand!));
      }
      return 'start' as const;
    },
    onTrial: async ({ page: p, trial }) => {
      if (!panelChecked && introHand && trial !== null && trial > 0) {
        panelChecked = true;
        await expect(p.getByTestId('tp-hand')).toHaveText(he.common.handShort[introHand]);
        expect(speedWords.has(norm(await p.getByTestId('tp-speed').innerText()))).toBe(true);
        expect(await dir()).toBe('rtl');
      } else {
        await p.waitForTimeout(100);
      }
    },
    onCheckpoint: async (p) => {
      checkpointChecked = true;
      expect(await dir()).toBe('rtl');
      await expect(p.getByTestId('btn-finish')).toHaveText(he.session.checkpoint.finish);
      await expect(p.getByTestId('checkpoint')).toContainText(he.session.checkpoint.bonusHint);
      await checkCheckpointLayout(p);
      return 'finish' as const;
    },
    deadlineMs: 10 * 60_000,
  });
  expect(run.end).toBe('done');
  expect(introChecked && panelChecked && checkpointChecked).toBe(true);

  await expect(page.getByTestId('done-thanks')).toHaveText(he.onboarding.done.thanks('Dana'));
  await expect(page.getByTestId('stat-accepted')).toHaveText(he.onboarding.done.recorded(29));
  expect(await numberIn(page.getByTestId('stat-accepted'))).toBe(29);
  expect(await dir()).toBe('rtl');
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
  expect(errors.consoleErrors, errors.consoleErrors.join('\n')).toEqual([]);

  const events = await readTimeline(page);
  const total = (events.find((e) => e.id === 'screen-done')?.t ?? 0) - (events.find((e) => e.id === 'screen-welcome')?.t ?? 0);
  printTimings('Simulated street visit in Hebrew (wall clock)', [{ phase: 'TOTAL (welcome → done)', ms: total }]);
});
