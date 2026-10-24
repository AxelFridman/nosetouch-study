/**
 * Participant journeys in simulation mode (?sim=1): the built-in simulator plays the participant, so the
 * real UI runs the v2 path on a desktop browser with a phone viewport (spec v2 §1, §3):
 *   welcome + consent → profile → session (How-to card → free trial → hand question → the hands-free
 *   blocks of tiers 1–3) → checkpoint → done → export.
 * Also: finishing early from the pause sheet (status ended_early), the researchers' setup screen
 * (?sim=1&setup=1) with a custom 2-trial plan, and their tap-review mode (Settings: review every
 * trial). Every export is validated against the schema; the street visit prints its wall-clock
 * timings. Only data-testid selectors drive the UI.
 * Run: npm run e2e (Playwright starts the Vite dev server, see playwright.config.ts).
 */
import { expect, silenceLog, test } from './fixtures';
import {
  checkCheckpointLayout,
  chooseTwoTrialProtocol,
  disableWebShare,
  exportBundle,
  finishFromPause,
  installTimeline,
  type IntroDecision,
  numberIn,
  onboard,
  openApp,
  printTimings,
  readTimeline,
  runSession,
  setToggle,
  streetTimings,
  tapFresh,
  watchErrors,
} from './helpers';
import { validate } from '../tests/helpers/schemaValidator';
import { APP_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION } from '../src/core/constants';
import type { ExportBundle, TrialRecord } from '../src/core/types';
import { en } from '../src/i18n/en';

/** Every record of the export is valid against the schema (as a whole and one by one). */
function expectSchemaValid(bundle: ExportBundle): void {
  const errors: string[] = [];
  const check = (def: Parameters<typeof validate>[0], value: unknown, what: string) => {
    for (const e of validate(def, value)) errors.push(`${what}: ${e}`);
  };
  check('ExportBundle', bundle, 'bundle');
  for (const p of bundle.participants) check('ParticipantProfile', p, `participant ${p.participant_id}`);
  for (const s of bundle.sessions) check('SessionRecord', s, `session ${s.session_id}`);
  for (const t of bundle.trials) check('TrialRecord', t, `trial ${t.trial_id}`);
  expect(errors, `schema ${SCHEMA_VERSION} errors:\n${errors.slice(0, 30).join('\n')}`).toEqual([]);
}

/** The accepted attempts, one per planned trial index (a planned trial is accepted at most once). */
function acceptedByIndex(bundle: ExportBundle): Map<number, TrialRecord> {
  const out = new Map<number, TrialRecord>();
  for (const t of bundle.trials.filter((x) => x.outcome.accepted)) {
    expect(out.has(t.trial_index), `trial ${t.trial_index} accepted twice`).toBe(false);
    out.set(t.trial_index, t);
  }
  return out;
}

test('street visit, simulated: How-to card, free trial, hand question, tiers 1–3 hands-free, checkpoint, done', async ({ page }) => {
  test.setTimeout(12 * 60_000);
  const errors = watchErrors(page);
  await disableWebShare(page);
  await installTimeline(page);

  await openApp(page, { lang: 'en' });
  await onboard(page);
  // the session opens on the How-to card with its illustration (spec v2 §1)
  await expect(page.getByTestId('howto-card')).toBeVisible();
  await expect(page.getByTestId('gesture-demo')).toBeVisible();
  await expect(page.getByTestId('btn-try')).toBeVisible();

  let sawCheckpoint = false;
  const run = await runSession(page, {
    handAnswer: 'right', // the simulated participant uses its usual phone hand (profile: right)
    onCheckpoint: async (p) => {
      sawCheckpoint = true;
      await checkCheckpointLayout(p);
      return 'finish' as const;
    },
    deadlineMs: 10 * 60_000,
  });
  expect(run.end).toBe('done');
  expect(sawCheckpoint, 'the checkpoint after tier 3').toBe(true);

  // done screen: thanks with the first name, the count, the participant code, the upload card
  await expect(page.getByTestId('done-thanks')).toHaveText(en.onboarding.done.thanks('Dana'));
  expect(await numberIn(page.getByTestId('stat-accepted'))).toBe(29);
  await expect(page.getByTestId('done-code')).toContainText(/P-[A-Z0-9]{6}/);
  await expect(page.getByTestId('upload-status')).toBeVisible();
  await expect(page.getByTestId('btn-next-person')).toBeVisible();

  const bundle = await exportBundle(page);
  expect(bundle.format).toBe('nose-touch-export');
  expectSchemaValid(bundle);

  expect(bundle.sessions).toHaveLength(1);
  const session = bundle.sessions[0];
  const required = session.protocol.plan.filter((b) => !b.optional).flatMap((b) => b.trials);
  expect(required).toHaveLength(29);
  expect(session).toMatchObject({
    status: 'completed',
    simulated: true,
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
  });
  expect(session.protocol).toMatchObject({ preset: 'street', protocol_version: PROTOCOL_VERSION, custom: null });
  expect(session.settings.review_mode).toBe('auto');
  expect(session.calibration?.method).toBe('implicit_v1');
  expect(session.progress.n_accepted).toBe(29);
  expect(session.participant.nickname).toBe('Dana Levi'); // names are uploaded/exported (config.json)

  // one accepted attempt for every planned trial of tiers 0–3, none of the bonus
  const accepted = acceptedByIndex(bundle);
  expect([...accepted.keys()].sort((a, b) => a - b)).toEqual(required.map((t) => t.trial_index));
  for (const t of accepted.values()) {
    expect(t.simulated).toBe(true);
    expect(t.outcome.status).toBe('complete');
    expect(t.streams.motion.t.length, `trial ${t.trial_index} samples`).toBeGreaterThan(50);
  }
  expect(bundle.trials.every((t) => t.simulated === true)).toBe(true);
  expect(bundle.trials.some((t) => t.outcome.note === 'hand_not_reported'), 'the placeholder copy was replaced by the answer').toBe(false);

  // the choice trial carries the REPORTED hand; every other trial the instructed one, auto-accepted
  const choice = accepted.get(0)!;
  expect(choice.condition).toMatchObject({ hand: 'right', hand_source: 'reported', orientation: 'free' });
  for (const [i, t] of accepted) {
    if (i === 0) continue;
    expect(t.condition.hand_source, `trial ${i}`).toBe('instructed');
    expect(t.outcome.note, `trial ${i}`).toBe('auto_accept');
  }
  const blocks = new Map(session.protocol.plan.map((b) => [b.block_index, b]));
  for (const t of accepted.values()) {
    const b = blocks.get(t.block_index)!;
    if (b.kind !== 'choice') expect(t.condition.hand, `trial ${t.trial_index}`).toBe(b.hand);
  }
  expect(new Set([...accepted.values()].map((t) => t.condition.hand))).toEqual(new Set(['left', 'right']));
  expect([...accepted.values()].some((t) => t.condition.variant === 'tilted_touch'), 'tier 3 always has tilted_touch').toBe(true);

  // the return of the arm was recorded on (nearly) every trial (spec v2 §3.5)
  const withReturn = [...accepted.values()].filter(
    (t) => typeof t.timing.return_onset_ms === 'number' && typeof t.timing.return_end_ms === 'number',
  );
  expect(withReturn.length, `${withReturn.length}/${accepted.size} trials with return timing`).toBeGreaterThanOrEqual(Math.ceil(0.8 * accepted.size));

  // the simulator path keeps ~60 Hz motion sampling with every v2 stream on (spec v2 §7); T5 needs ≥ 50 Hz
  const hz = [...accepted.values()].map((t) => t.quality.motion_hz ?? 0).sort((a, b) => a - b);
  const medianHz = hz[Math.floor(hz.length / 2)];
  const gaps = [...accepted.values()].reduce((n, t) => n + t.quality.gaps_over_50ms, 0);
  console.log(`motion sampling in the browser: median ${medianHz.toFixed(1)} Hz, min ${hz[0].toFixed(1)} Hz; gaps > 50 ms: ${gaps}`);
  expect(medianHz).toBeGreaterThanOrEqual(50);
  expect([...accepted.values()].some((t) => (t.streams.quaternion_abs?.t.length ?? 0) > 0), 'absolute quaternion stream (Android-like)').toBe(true);

  // never a sound, never an error
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors, `uncaught errors in the page:\n${errors.pageErrors.join('\n')}`).toEqual([]);
  expect(errors.consoleErrors, `console errors:\n${errors.consoleErrors.join('\n')}`).toEqual([]);

  const rows = streetTimings(await readTimeline(page), session.protocol.plan);
  printTimings(
    `Simulated street visit (wall clock; ${accepted.size} accepted of ${bundle.trials.length} attempts; pauses: ${run.pauses.join(', ') || 'none'})`,
    rows,
  );
});

test('finishing from the pause sheet after the basics ends the session as ended_early', async ({ page }) => {
  test.setTimeout(8 * 60_000);
  const errors = watchErrors(page);
  await disableWebShare(page);
  await openApp(page, { lang: 'en' });
  await onboard(page);

  // play the free trial and tier 1; stop at the first intro after the basics
  const run = await runSession(page, {
    onIntro: async ({ page: p }) => ((await p.getByTestId('intro-encourage').isVisible()) ? 'stop' : 'start') as IntroDecision,
  });
  expect(run.end).toBe('stopped');
  await expect(page.getByTestId('intro-encourage')).toHaveText(en.session.intro.basicsDone);

  // the header's Pause opens the pause sheet; Finish now ends the visit
  await tapFresh(page, page.getByTestId('btn-session-exit'));
  await expect(page.getByTestId('pause-sheet')).toBeVisible();
  await expect(page.getByTestId('pause-reason')).toHaveAttribute('data-reason', 'paused');
  await finishFromPause(page);

  await expect(page.getByTestId('screen-done')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('screen-done')).toContainText(en.onboarding.done.endedEarly);
  expect(await numberIn(page.getByTestId('stat-accepted'))).toBe(9);

  const bundle = await exportBundle(page);
  expectSchemaValid(bundle);
  const [session] = bundle.sessions;
  expect(session.status).toBe('ended_early');
  expect(session.ended_at).toEqual(expect.any(String));
  expect(session.progress.n_accepted).toBe(9);
  const basics = session.protocol.plan.filter((b) => (b.tier ?? 0) <= 1).flatMap((b) => b.trials.map((t) => t.trial_index));
  expect([...acceptedByIndex(bundle).keys()].sort((a, b) => a - b)).toEqual(basics);
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test('researchers: ?sim=1&setup=1 opens the setup screen; a custom 2-trial plan runs hands-free', async ({ page }) => {
  const errors = watchErrors(page);
  await disableWebShare(page);
  await openApp(page, { lang: 'en', setup: true });
  await expect(page.getByTestId('warn-setup')).toBeVisible();
  await onboard(page, { until: 'setup' });

  await chooseTwoTrialProtocol(page);
  await expect(page.getByTestId('setup-estimate')).toBeVisible();
  await tapFresh(page, page.getByTestId('btn-start-session'));
  // no choice block in a custom plan: the session opens on a block intro, not the How-to card
  await expect(page.getByTestId('block-intro')).toBeVisible();
  const run = await runSession(page);
  expect(run.end).toBe('done');
  expect(run.intros).toHaveLength(2); // one block per hand

  expect(await numberIn(page.getByTestId('stat-accepted'))).toBe(2);
  const bundle = await exportBundle(page);
  expectSchemaValid(bundle);
  const [session] = bundle.sessions;
  expect(session.status).toBe('completed');
  expect(session.protocol.preset).toBe('custom');
  expect(session.settings.review_mode).toBe('auto');
  const accepted = bundle.trials.filter((t) => t.outcome.accepted && t.outcome.status === 'complete');
  expect(accepted).toHaveLength(2);
  expect(new Set(accepted.map((t) => t.condition.hand))).toEqual(new Set(['left', 'right']));
  for (const t of accepted) {
    expect(t.condition.hand_source).toBe('instructed');
    expect(t.streams.motion.t.length).toBeGreaterThan(50);
  }
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});

test('researchers: the tap-review mode (Settings → review every trial) still runs a session', async ({ page }) => {
  const errors = watchErrors(page);
  await disableWebShare(page);
  await openApp(page, { lang: 'en' });
  // both researcher switches in Settings: review every trial, and the setup screen after the profile
  await page.getByTestId('btn-menu-settings').click();
  await setToggle(page.getByTestId('chk-review-each-trial'), true);
  await setToggle(page.getByTestId('chk-setup-mode'), true);
  await page.getByTestId('btn-back').click();
  await expect(page.getByTestId('screen-welcome')).toBeVisible();
  await onboard(page, { until: 'setup' });
  await chooseTwoTrialProtocol(page);
  await tapFresh(page, page.getByTestId('btn-start-session'));

  // block intro → Ready → recording → review (Yes, next) → break → … → done
  await expect(page.getByTestId('block-intro')).toBeVisible();
  const run = await runSession(page);
  expect(run.end).toBe('done');
  expect(await numberIn(page.getByTestId('stat-accepted'))).toBe(2);

  const bundle = await exportBundle(page);
  expectSchemaValid(bundle);
  const [session] = bundle.sessions;
  expect(session.settings.review_mode).toBe('tap');
  expect(session.status).toBe('completed');
  const accepted = bundle.trials.filter((t) => t.outcome.accepted);
  expect(accepted).toHaveLength(2);
  for (const t of accepted) expect(t.outcome.note).toBeNull(); // the answered review dropped 'review_interrupted'
  expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
  expect(errors.pageErrors).toEqual([]);
  expect(errors.consoleErrors).toEqual([]);
});
