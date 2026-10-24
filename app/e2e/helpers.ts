/**
 * Shared Playwright helpers for the simulated participant (?sim=1). Only data-testid selectors are
 * used (never UI text), so every helper works in any UI language.
 *
 * The flows are LOOPS that act on whatever screen is visible, with a deadline, instead of fixed step
 * lists: onboard() handles welcome + consent, the profile, the sensor repair screen and (with
 * ?setup=1) the researchers' setup screen; runSession() taps through the How-to card, the hand
 * question, block intros, the checkpoint and the pause sheet (and the tap-review screens of the
 * researcher mode) until the done screen — the hands-free trials in between need no tap at all.
 * Hooks let a spec stop or decide at block intros, the checkpoint and the pause sheet.
 */
import { readFile } from 'node:fs/promises';
import type { Locator, Page } from '@playwright/test';
import { expect } from './fixtures';
import { TAP_GUARD_MS } from '../src/app/state';
import { COLLECTED_ORIENTATIONS, ODD_VARIANTS } from '../src/core/constants';
import type { ExportBundle, Hand, HandednessSelf, PhoneHand, Sex, Speed, StartPosition } from '../src/core/types';

export const SPEEDS: Speed[] = ['natural', 'fast', 'slow'];
export const STARTS: StartPosition[] = ['natural', 'shoulder', 'center'];
/** A control of a screen that has just appeared ignores taps this long (the app's double-tap guard). */
export const TAP_DELAY_MS = TAP_GUARD_MS + 150;

// ───────────────────────────── small actions ─────────────────────────────

/**
 * Clicks a button of a step that has just appeared. The app ignores taps within TAP_GUARD_MS of a
 * step change (its double-tap guard), and the session's buttons that appear without a tap stay
 * disabled for its arm delay; Playwright is faster than any participant.
 */
export async function tapFresh(page: Page, button: Locator, timeout = 30_000): Promise<void> {
  await expect(button).toBeEnabled({ timeout });
  await page.waitForTimeout(TAP_DELAY_MS);
  await button.click();
}

/** The checkbox/radio behind a tile or label: the element itself if it is an input, else its inner input. */
export async function inputOf(el: Locator): Promise<Locator> {
  const tag = await el.evaluate((node) => node.tagName.toLowerCase());
  return tag === 'input' ? el : el.locator('input').first();
}

/** Ticks or unticks a Toggle (label with a checkbox) so that it ends up `checked`. */
export async function setToggle(toggle: Locator, checked: boolean): Promise<void> {
  const input = await inputOf(toggle);
  if ((await input.isChecked()) !== checked) await toggle.click();
  await expect(input).toBeChecked({ checked });
}

/** Makes the checked tiles of a MultiChoice group exactly `wanted`, clicking only what must change. */
export async function setTiles(page: Page, prefix: string, all: readonly string[], wanted: readonly string[]): Promise<void> {
  for (const value of wanted) {
    // a wanted tile that does not exist would leave the group in another state than the spec asks for
    await expect(page.getByTestId(`${prefix}-${value}`), `tile ${prefix}-${value}`).toHaveCount(1);
  }
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

/** Collects uncaught page errors and console errors (call before the first goto). */
export function watchErrors(page: Page): { pageErrors: string[]; consoleErrors: string[] } {
  const out = { pageErrors: [] as string[], consoleErrors: [] as string[] };
  page.on('pageerror', (e) => out.pageErrors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') out.consoleErrors.push(m.text());
  });
  return out;
}

// ───────────────────────────── what is on screen ─────────────────────────────

/**
 * The cards and screens the loops recognise, most specific first. The session's cards all sit in a
 * `screen-session`; the pause sheet (session-exit) replaces the card under it.
 */
export const VIEW_IDS = [
  'screen-boot-failed',
  'screen-done',
  'session-exit',
  'hand-report',
  'checkpoint',
  'howto-card',
  'block-intro',
  'trial-panel',
  'trial-review',
  'trial-failed',
  'block-break',
  'trial-ready',
  'btn-continue-session', // a paused chain without its sheet (the sheet's own button is inside session-exit)
  'screen-setup',
  'screen-sensorcheck',
  'screen-profile',
  'screen-welcome',
] as const;
export type ViewId = (typeof VIEW_IDS)[number];

export interface ViewInfo {
  id: ViewId;
  /** trial-panel: data-phase */
  phase: string | null;
  /** trial-panel: data-trial-index */
  trial: number | null;
  /** block-intro: data-block-index */
  block: number | null;
  /** session-exit: the pause reason (data-reason) */
  reason: string | null;
}

/** What is on screen now (null: nothing recognisable yet, e.g. the loading spinner). */
export async function currentView(page: Page): Promise<ViewInfo | null> {
  return page.evaluate((ids) => {
    const shown = (el: Element | null): el is HTMLElement => {
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    };
    for (const id of ids) {
      const el = document.querySelector(`[data-testid="${id}"]`);
      if (!shown(el)) continue;
      const num = (v: string | null) => (v === null || v === '' ? null : Number(v));
      return {
        id,
        phase: el.getAttribute('data-phase'),
        trial: num(el.getAttribute('data-trial-index')),
        block: num(el.getAttribute('data-block-index')),
        reason: id === 'session-exit' ? (document.querySelector('[data-testid="pause-reason"]')?.getAttribute('data-reason') ?? null) : null,
      };
    }
    return null;
  }, VIEW_IDS as unknown as string[]) as Promise<ViewInfo | null>;
}

function viewKey(v: ViewInfo | null): string {
  return v ? `${v.id}|${v.trial ?? ''}|${v.block ?? ''}|${v.reason ?? ''}` : 'none';
}

/** Waits until the view differs from `before` (after a tap), at most `timeout` ms. */
async function waitForViewChange(page: Page, before: ViewInfo | null, timeout = 15_000): Promise<void> {
  const key = viewKey(before);
  await expect.poll(async () => viewKey(await currentView(page)), { timeout, intervals: [50, 100, 200] }).not.toBe(key);
}

/**
 * Taps button `id` of the view `v` once it is usable (tap guard / arm delay), then waits for the view
 * to change. Returns false without tapping when the view moved on by itself first (e.g. a save that
 * finished, or a tap of the spec that the loop had not seen yet), so a loop never waits on a button
 * that is gone.
 */
async function tapInView(page: Page, id: string, v: ViewInfo, timeout = 30_000): Promise<boolean> {
  const button = page.getByTestId(id);
  const key = viewKey(v);
  const moved = async () => viewKey(await currentView(page)) !== key;
  const usable = async () => (await button.count()) > 0 && (await button.isEnabled());
  await expect
    .poll(async () => ((await moved()) ? 'moved' : (await usable()) ? 'usable' : 'wait'), { timeout, intervals: [50, 100, 200] })
    .not.toBe('wait');
  if (await moved()) return false;
  await page.waitForTimeout(TAP_DELAY_MS);
  if (await moved()) return false;
  await button.click();
  await waitForViewChange(page, v);
  return true;
}

// ───────────────────────────── onboarding ─────────────────────────────

export interface ProfileAnswers {
  name?: string;
  age?: number;
  sex?: Sex;
  handed?: HandednessSelf;
  phoneHand?: PhoneHand;
  height?: number | null;
}

/** A right-handed simulated participant who holds the phone in the right hand. */
export const DEFAULT_PROFILE: Required<ProfileAnswers> = {
  name: 'Dana Levi',
  age: 30,
  sex: 'female',
  handed: 'right',
  phoneHand: 'right',
  height: 168,
};

/** Fills the "About you" form; only fields whose test id exists are touched. */
export async function fillProfile(page: Page, answers: ProfileAnswers = DEFAULT_PROFILE): Promise<void> {
  const a = { ...DEFAULT_PROFILE, ...answers };
  const fill = async (id: string, value: string) => {
    const el = page.getByTestId(id);
    if ((await el.count()) > 0) await el.fill(value);
  };
  const choose = async (id: string) => {
    const el = page.getByTestId(id);
    if ((await el.count()) > 0) await el.click();
  };
  await fill('in-name', a.name);
  await fill('in-age', String(a.age));
  await choose(`opt-sex-${a.sex}`);
  await choose(`opt-handed-${a.handed}`);
  await choose(`opt-phonehand-${a.phoneHand}`);
  if (a.height !== null) await fill('in-height', String(a.height));
}

export interface OpenOptions {
  lang?: 'en' | 'he';
  /** ?setup=1: the profile's Continue opens the researchers' setup screen */
  setup?: boolean;
}

/** Opens the app in simulation mode (?sim=1) and waits for the welcome screen. */
export async function openApp(page: Page, opts: OpenOptions = {}): Promise<void> {
  const params = new URLSearchParams({ sim: '1', lang: opts.lang ?? 'en' });
  if (opts.setup) params.set('setup', '1');
  await page.goto(`./?${params.toString()}`);
  await expect(page.getByTestId('screen-welcome')).toBeVisible({ timeout: 30_000 });
}

export interface OnboardOptions {
  /** where to stop: the session (How-to card or a block intro) or, with ?setup=1, the setup screen */
  until?: 'session' | 'setup';
  profile?: ProfileAnswers;
  deadlineMs?: number;
}

/**
 * From the welcome screen to the session (or the setup screen), acting on whatever is visible:
 * welcome (consent checkbox + Start), profile (answers + Continue), the sensor repair screen
 * (permission + Continue) and, unless it is the target, the setup screen (its default preset).
 */
export async function onboard(page: Page, opts: OnboardOptions = {}): Promise<void> {
  const until = opts.until ?? 'session';
  const deadline = Date.now() + (opts.deadlineMs ?? 90_000);
  let filled = false;
  let last: ViewInfo | null = null;
  while (Date.now() < deadline) {
    const v = await currentView(page);
    last = v;
    switch (v?.id) {
      case 'screen-welcome': {
        await setToggle(page.getByTestId('chk-consent'), true);
        await tapInView(page, 'btn-start', v);
        break;
      }
      case 'screen-profile': {
        if (!filled) {
          await fillProfile(page, opts.profile);
          filled = true;
        }
        await tapInView(page, 'btn-save-profile', v); // Continue waits for the background sensor check (≤ 3 s)
        break;
      }
      case 'screen-sensorcheck': {
        const ask = page.getByTestId('btn-request-permission');
        if (await ask.isVisible().catch(() => false)) await ask.click();
        await tapInView(page, 'btn-continue', v);
        break;
      }
      case 'screen-setup': {
        if (until === 'setup') return;
        await tapInView(page, 'btn-start-session', v);
        break;
      }
      case 'howto-card':
      case 'block-intro':
      case 'trial-ready':
      case 'checkpoint':
        if (until === 'session') return;
        throw new Error(`onboard: reached the session (${v.id}) but expected the setup screen (open with ?setup=1)`);
      case 'screen-boot-failed':
        throw new Error('onboard: the app could not start (screen-boot-failed)');
      default:
        await page.waitForTimeout(200);
    }
  }
  throw new Error(`onboard: no ${until} screen before the deadline; last view: ${JSON.stringify(last)}`);
}

// ───────────────────────────── the session ─────────────────────────────

export type IntroDecision = 'start' | 'stop';
export type CheckpointDecision = 'finish' | 'bonus' | 'stop';
export type PauseDecision = 'continue' | 'finish' | 'stop';

export interface RunSessionOptions {
  /** hand question after the free trial (default: the simulated participant's phone hand, right) */
  handAnswer?: Hand | 'unsure';
  /** called at every block intro before its Start is tapped */
  onIntro?: (info: { block: number | null; page: Page }) => Promise<IntroDecision> | IntroDecision;
  /** called at the checkpoint (default: finish) */
  onCheckpoint?: (page: Page) => Promise<CheckpointDecision> | CheckpointDecision;
  /**
   * called when the pause sheet is open (default: continue — a simulated trial may fail twice or the
   * chain may pause; the journey goes on and the reason is logged)
   */
  onPause?: (reason: string | null, page: Page) => Promise<PauseDecision> | PauseDecision;
  /** called while a trial records (e.g. to press the pause control) */
  onTrial?: (info: { trial: number | null; phase: string | null; page: Page }) => Promise<void> | void;
  deadlineMs?: number;
}

export interface SessionRunReport {
  /** how the loop ended */
  end: 'done' | 'stopped';
  /** the view the loop stopped at */
  last: ViewInfo | null;
  /** reasons of every pause sheet that was seen (data-reason) */
  pauses: string[];
  /** distinct trial indices seen on the live panel, in order */
  trials: number[];
  /** block indices of the intros that were tapped */
  intros: number[];
}

/**
 * Plays the session until the done screen (or until a hook says 'stop'), acting on whatever is
 * visible: How-to card (Try it once), hand question, block intros, checkpoint, pause sheet, and the
 * tap-review screens (Ready / Next / Try again / next block) of the researcher mode. Hands-free trials
 * need nothing: the loop only watches them.
 */
export async function runSession(page: Page, opts: RunSessionOptions = {}): Promise<SessionRunReport> {
  const deadline = Date.now() + (opts.deadlineMs ?? 8 * 60_000);
  const report: SessionRunReport = { end: 'done', last: null, pauses: [], trials: [], intros: [] };
  const tap = async (id: string, v: ViewInfo): Promise<boolean> => tapInView(page, id, v);
  while (Date.now() < deadline) {
    const v = await currentView(page);
    report.last = v;
    switch (v?.id) {
      case 'screen-done':
        report.end = 'done';
        return report;
      case 'howto-card':
        await tap('btn-try', v);
        break;
      case 'hand-report':
        await tap(`btn-hand-${opts.handAnswer ?? 'right'}`, v);
        break;
      case 'block-intro': {
        if (opts.onIntro && (await opts.onIntro({ block: v.block, page })) === 'stop') {
          report.end = 'stopped';
          return report;
        }
        if ((await tap('btn-block-start', v)) && v.block !== null) report.intros.push(v.block);
        break;
      }
      case 'checkpoint': {
        const d = opts.onCheckpoint ? await opts.onCheckpoint(page) : 'finish';
        if (d === 'stop') {
          report.end = 'stopped';
          return report;
        }
        await tap(d === 'bonus' ? 'btn-bonus' : 'btn-finish', v);
        break;
      }
      case 'session-exit': {
        report.pauses.push(v.reason ?? '?');
        const d = opts.onPause ? await opts.onPause(v.reason, page) : 'continue';
        if (d === 'stop') {
          report.end = 'stopped';
          return report;
        }
        if (d === 'continue') console.warn(`runSession: the pause sheet opened (${v.reason}); continuing`);
        await tap(d === 'finish' ? 'btn-finish-now' : 'btn-continue-session', v);
        break;
      }
      case 'btn-continue-session':
        // a paused chain whose sheet was closed: open it again
        await tap('btn-continue-session', v);
        break;
      case 'trial-panel':
        if (v.trial !== null && report.trials[report.trials.length - 1] !== v.trial) report.trials.push(v.trial);
        if (opts.onTrial) await opts.onTrial({ trial: v.trial, phase: v.phase, page });
        else await page.waitForTimeout(100);
        break;
      // researcher mode (settings: review every trial)
      case 'trial-ready':
        await tap('btn-ready', v);
        break;
      case 'trial-review':
        await tap('btn-next', v);
        break;
      case 'trial-failed':
        await tap('btn-retry', v);
        break;
      case 'block-break':
        await tap('btn-next-block', v);
        break;
      case 'screen-boot-failed':
        throw new Error('runSession: the app could not start (screen-boot-failed)');
      default:
        await page.waitForTimeout(150);
    }
  }
  throw new Error(`runSession: no done screen before the deadline; last view: ${JSON.stringify(report.last)}`);
}

// ───────────────────────────── researchers' setup screen (?setup=1) ─────────────────────────────

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
  await setTiles(page, 'opt-orient', COLLECTED_ORIENTATIONS, choice.orientations);
  await setTiles(page, 'opt-speed', SPEEDS, choice.speeds);
  await setTiles(page, 'opt-start', STARTS, choice.starts);
  await setTiles(page, 'opt-odd', ODD_VARIANTS, choice.odd ?? []);
  await page.getByTestId('in-reps').fill('1');
  const free = page.getByTestId('chk-free');
  if ((await free.count()) > 0) await setToggle(free, false);
}

/** Custom protocol: upright_toward × natural speed × natural start × 1 rep → one trial per hand. */
export async function chooseTwoTrialProtocol(page: Page): Promise<void> {
  await chooseCustomProtocol(page, { orientations: ['upright_toward'], speeds: ['natural'], starts: ['natural'] });
}

// ───────────────────────────── results ─────────────────────────────

/** Done screen → "Save / share my data" → the downloaded export (needs disableWebShare before the goto). */
export async function exportBundle(page: Page): Promise<ExportBundle> {
  const button = page.getByTestId('btn-export');
  await expect(button).toBeEnabled();
  const downloadPromise = page.waitForEvent('download');
  await button.click();
  const download = await downloadPromise;
  return JSON.parse(await readFile(await download.path(), 'utf8')) as ExportBundle;
}

/** The first integer in an element's text (e.g. "29 movements recorded"). */
export async function numberIn(el: Locator): Promise<number> {
  const text = (await el.innerText()).trim();
  const m = text.match(/\d+/);
  if (!m) throw new Error(`no number in "${text}"`);
  return Number(m[0]);
}

// ───────────────────────────── layout ─────────────────────────────

/** The element is rendered entirely inside the viewport (nothing to scroll to reach it). */
export async function expectInViewport(page: Page, el: Locator, what: string): Promise<void> {
  await expect(el, what).toBeVisible();
  const b = await el.boundingBox();
  const vp = page.viewportSize();
  if (!b || !vp) throw new Error(`${what}: not rendered`);
  expect(b.x, `${what}: left edge`).toBeGreaterThanOrEqual(-0.5);
  expect(b.y, `${what}: top edge`).toBeGreaterThanOrEqual(-0.5);
  expect(b.x + b.width, `${what}: right edge vs viewport ${vp.width}`).toBeLessThanOrEqual(vp.width + 0.5);
  expect(b.y + b.height, `${what}: bottom edge vs viewport ${vp.height}`).toBeLessThanOrEqual(vp.height + 0.5);
}

/** `item` lies inside the viewport, entirely above the top edge of `button` (a sticky footer button). */
export async function expectAbove(page: Page, item: Locator, button: Locator, what: string): Promise<void> {
  await expectInViewport(page, item, what);
  const a = await item.boundingBox();
  const b = await button.boundingBox();
  if (!a || !b) throw new Error(`${what}: not rendered`);
  expect(a.y + a.height, `${what}: bottom edge vs the button's top`).toBeLessThanOrEqual(b.y + 0.5);
}

/** The page never scrolls sideways (a phone screen must not pan horizontally). */
export async function expectNoHorizontalScroll(page: Page, what: string): Promise<void> {
  const { scroll, width } = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    width: window.innerWidth,
  }));
  expect(scroll, `${what}: page width ${scroll} > viewport ${width}`).toBeLessThanOrEqual(width + 1);
}

/** Small phones of the layout checks: a small Android phone, the visible viewport of an iPhone 12–15 in Safari. */
export const SMALL_VIEWPORTS = [
  { width: 360, height: 640 },
  { width: 390, height: 664 },
] as const;

/** The checkpoint card fits small phones: its text and both buttons without scrolling (viewport restored after). */
export async function checkCheckpointLayout(page: Page): Promise<void> {
  const original = page.viewportSize();
  for (const vp of SMALL_VIEWPORTS) {
    await page.setViewportSize(vp);
    await page.evaluate(() => window.scrollTo(0, 0));
    const what = `checkpoint ${vp.width}×${vp.height}`;
    await expectInViewport(page, page.getByTestId('btn-finish'), `${what}: Finish`);
    await expectInViewport(page, page.getByTestId('btn-bonus'), `${what}: Bonus`);
    await expectInViewport(page, page.getByTestId('checkpoint').locator('.checkpoint-body'), `${what}: text`);
    await expectNoHorizontalScroll(page, what);
  }
  if (original) await page.setViewportSize(original);
}

/** The page does not scroll at all vertically (the live panel: arm's length, no free hand to scroll). */
export async function expectNoVerticalScroll(page: Page, what: string): Promise<void> {
  const { scroll, height } = await page.evaluate(() => ({
    scroll: document.documentElement.scrollHeight,
    height: window.innerHeight,
  }));
  expect(scroll, `${what}: page height ${scroll} > viewport ${height}`).toBeLessThanOrEqual(height + 1);
}

// ───────────────────────────── wall-clock timeline ─────────────────────────────

export interface TimelineEvent {
  /** ms since the page's navigation start (performance.now()) */
  t: number;
  id: string;
  phase: string | null;
  trial: number | null;
  block: number | null;
  reason: string | null;
}

/**
 * Records, in the page, when each screen/card appears and every live-panel phase change (data-phase,
 * e.g. 'armed' = GO), into window.__e2eTimeline. Call before the first goto.
 */
export async function installTimeline(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const WATCH = [
      'screen-welcome',
      'screen-profile',
      'screen-sensorcheck',
      'screen-setup',
      'howto-card',
      'hand-report',
      'block-intro',
      'trial-panel',
      'checkpoint',
      'session-exit',
      'screen-done',
    ];
    const events: Array<Record<string, unknown>> = [];
    (window as unknown as { __e2eTimeline: unknown }).__e2eTimeline = events;
    const last = new Map<Element, string>();
    const num = (v: string | null) => (v === null || v === '' ? null : Number(v));
    const note = (el: Element) => {
      const id = el.getAttribute('data-testid');
      if (!id || !WATCH.includes(id)) return;
      const rec = {
        t: performance.now(),
        id,
        phase: el.getAttribute('data-phase'),
        trial: num(el.getAttribute('data-trial-index')),
        block: num(el.getAttribute('data-block-index')),
        reason: id === 'session-exit' ? (el.querySelector('[data-testid="pause-reason"]')?.getAttribute('data-reason') ?? null) : null,
      };
      const key = `${rec.phase}|${rec.trial}|${rec.block}|${rec.reason}`;
      if (last.get(el) === key) return;
      last.set(el, key);
      events.push(rec);
    };
    const scan = (node: Node) => {
      if (!(node instanceof Element)) return;
      note(node);
      for (const el of node.querySelectorAll('[data-testid]')) note(el);
    };
    new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'attributes') {
          const el = m.target as Element;
          note(el);
          // a reused intro / pause sheet: its inner attributes changed
          const owner = el.closest('[data-testid="session-exit"]');
          if (owner) note(owner);
        } else {
          m.addedNodes.forEach(scan);
        }
      }
    }).observe(document, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['data-phase', 'data-trial-index', 'data-block-index', 'data-reason', 'data-testid'],
    });
  });
}

/** The events recorded by installTimeline, in order. */
export async function readTimeline(page: Page): Promise<TimelineEvent[]> {
  return page.evaluate(() => ((window as unknown as { __e2eTimeline?: unknown[] }).__e2eTimeline ?? []) as never);
}

/** Seconds with one decimal, for the timing tables printed by the journeys. */
export function secs(ms: number | null | undefined): string {
  return typeof ms === 'number' && Number.isFinite(ms) ? `${(ms / 1000).toFixed(1)} s` : '–';
}

export interface PhaseTiming {
  phase: string;
  ms: number | null;
}

/**
 * Wall-clock durations of the phases of a street visit from the recorded timeline, with the tiers of
 * the session's plan (block_index → tier): onboarding, How-to card + free trial + hand question,
 * each tier (from its first intro to the next tier's intro or the checkpoint), checkpoint → done,
 * the time to the first GO from opening the link, and the total.
 */
export function streetTimings(events: TimelineEvent[], plan: { block_index: number; tier?: number; optional?: boolean }[]): PhaseTiming[] {
  const first = (pred: (e: TimelineEvent) => boolean): number | null => events.find(pred)?.t ?? null;
  const span = (a: number | null, b: number | null): number | null => (a !== null && b !== null ? b - a : null);
  const tierOf = new Map(plan.map((b) => [b.block_index, b.tier ?? null]));
  const welcome = first((e) => e.id === 'screen-welcome');
  const profile = first((e) => e.id === 'screen-profile');
  const howto = first((e) => e.id === 'howto-card');
  const firstGo = first((e) => e.id === 'trial-panel' && e.phase === 'armed');
  const checkpoint = first((e) => e.id === 'checkpoint');
  const done = first((e) => e.id === 'screen-done');
  const tierStart = (tier: number) => first((e) => e.id === 'block-intro' && e.block !== null && tierOf.get(e.block) === tier);
  const tiers = [...new Set(plan.filter((b) => !b.optional).map((b) => b.tier ?? 0))].filter((t) => t > 0).sort((a, b) => a - b);
  const rows: PhaseTiming[] = [
    { phase: 'page load → welcome screen', ms: welcome },
    { phase: 'welcome + consent (Start)', ms: span(welcome, profile) },
    { phase: 'profile (Continue) → How-to card', ms: span(profile, howto) },
    { phase: 'How-to card → first GO', ms: span(howto, firstGo) },
    { phase: 'free trial + hand question (tier 0)', ms: span(howto, tiers.length ? tierStart(tiers[0]) : checkpoint) },
  ];
  tiers.forEach((tier, i) => {
    const end = i + 1 < tiers.length ? tierStart(tiers[i + 1]) : (checkpoint ?? done);
    const n = plan.filter((b) => (b.tier ?? 0) === tier).length;
    rows.push({ phase: `tier ${tier} (${n} blocks)`, ms: span(tierStart(tier), end) });
  });
  rows.push(
    { phase: 'checkpoint → done (Finish)', ms: span(checkpoint, done) },
    { phase: 'opening the link → first GO', ms: firstGo },
    { phase: 'TOTAL (welcome → done)', ms: span(welcome, done) },
  );
  return rows;
}

/** Prints a timing table to the test output. */
export function printTimings(title: string, rows: PhaseTiming[]): void {
  const width = Math.max(...rows.map((r) => r.phase.length));
  console.log([`\n${title}`, ...rows.map((r) => `  ${r.phase.padEnd(width)}  ${secs(r.ms).padStart(8)}`)].join('\n'));
}
