/**
 * Live uploads (spec v2 §8): with the json_post mode, every saved trial and every session update is
 * sent at once, during the session, and the done screen shows the outbox state ("All data uploaded").
 * The upload endpoint is a same-origin path whose requests the test answers itself (page.route), so
 * nothing leaves the computer. Simulation mode; data-testid selectors only.
 */
import { expect, silenceLog, test } from './fixtures';
import { chooseTwoTrialProtocol, onboard, openApp, runSession, tapFresh, watchErrors } from './helpers';
import { validate } from '../tests/helpers/schemaValidator';
import type { SessionRecord, TrialRecord, UploadEnvelope } from '../src/core/types';

const ENDPOINT = '/api/ingest';

test('json_post: every trial and session update is uploaded at once; the done screen says all is sent', async ({ page }) => {
  const errors = watchErrors(page);
  const received: { at: number; env: UploadEnvelope }[] = [];
  await page.route(`**${ENDPOINT}`, async (route) => {
    const req = route.request();
    if (req.method() !== 'POST') return route.fulfill({ status: 204 });
    const env = JSON.parse(req.postData() ?? 'null') as UploadEnvelope;
    received.push({ at: Date.now(), env });
    const id = env.kind === 'session' ? (env.record as SessionRecord).session_id : (env.record as TrialRecord).trial_id;
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, kind: env.kind, id }) });
  });

  // researcher settings: upload with json_post to the test's own endpoint
  await openApp(page, { lang: 'en', setup: true });
  await page.getByTestId('btn-menu-settings').click();
  await page.getByTestId('opt-uploadmode-json_post').click();
  await page.getByTestId('in-endpoint').fill(ENDPOINT);
  await page.getByTestId('btn-save-settings').click();
  await page.getByTestId('btn-back').click();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();

  await onboard(page, { until: 'setup' });
  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));
  const trialsBy = () => received.filter((r) => r.env.kind === 'trial').length;
  let firstTrialUploadedDuringSession = false;
  let intros = 0;
  const run = await runSession(page, {
    onIntro: async () => {
      intros += 1;
      if (intros === 2) {
        // the first block's trial is sent right after it was saved (the queue's own timer runs only
        // every 60 s), while the participant is still in the session
        await expect.poll(trialsBy, { timeout: 5_000 }).toBeGreaterThanOrEqual(1);
        firstTrialUploadedDuringSession = true;
      }
      return 'start' as const;
    },
  });
  expect(run.end).toBe('done');
  expect(firstTrialUploadedDuringSession).toBe(true);

  // the done screen shows the live outbox state: everything sent
  await expect(page.getByTestId('upload-status').locator('[data-upload]')).toHaveAttribute('data-upload', 'all_sent', { timeout: 20_000 });

  // what arrived: valid envelopes, the session (finished, with the name) and every attempt
  for (const { env } of received) expect(validate('UploadEnvelope', env), `envelope ${env.kind}`).toEqual([]);
  const sessions = received.filter((r) => r.env.kind === 'session').map((r) => r.env.record as SessionRecord);
  expect(sessions.length).toBeGreaterThanOrEqual(3); // created, after each trial, finished
  const last = sessions[sessions.length - 1];
  expect(last.status).toBe('completed');
  expect(last.settings.review_mode).toBe('auto');
  expect(last.participant.nickname).toBe('Dana Levi'); // names are uploaded (config upload_nickname)
  const trials = new Map(received.filter((r) => r.env.kind === 'trial').map((r) => [(r.env.record as TrialRecord).trial_id, r.env.record as TrialRecord]));
  expect([...trials.values()].filter((t) => t.outcome.accepted)).toHaveLength(2);
  expect(last.progress.n_attempts).toBe(trials.size);
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, errors.pageErrors.join('\n')).toEqual([]);
  expect(errors.consoleErrors, errors.consoleErrors.join('\n')).toEqual([]);
});
