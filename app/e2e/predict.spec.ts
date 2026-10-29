/**
 * Predict mode (?predict=1, docs/stage3/APP_SPEC.md section 7.2) at phone sizes: a Pixel 7 (the project's
 * device) and an iPhone viewport with an iPhone user agent (Chromium either way; real iOS Safari and
 * Android Chrome stay manual checks: the permission prompt, the wake lock and the vibration).
 *
 *  - the desktop demo (&sim=1): a known answer through the same live path as the sensors, with no
 *    config.json request, no IndexedDB and no sound;
 *  - injected DeviceMotion events (the real-sensor path, WebKit sign);
 *  - the permission, denied and no-sensor screens;
 *  - every screen state (dev-only &preview=<state>) without horizontal scroll, and Hebrew right to left
 *    with the hand bar still LEFT on the left;
 *  - a committed synthetic fixture injected as DeviceMotion events with its exact sample times: the page
 *    shows the P(right) of the same engine replayed in Node (scripts/predictReplay.ts), for the driver and
 *    for every row of the compare view.
 *
 * The page runs the real engine (detect/ + models/ + pipeline/, data-engine="real"). The known-answer checks
 * drive with &model=m4 (lane m4's classifier, the one the detector and the stopping rule were fitted for; also
 * the demo's default): the final CNN reads simulated touches with the wrong sign (the exported synthetic
 * fixtures syn01..syn05: P(right) 0.2 % for a simulated right hand, 97-100 % for simulated left hands), m4,
 * PHYS6 and yaw do not.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import type { Page } from '@playwright/test';
import type { MotionSample } from '../src/core/types';
import { makeDemoScript } from '../src/predict/sim/demoScript';
import { formatPercent } from '../src/predict/ui/format';
import { expect, silenceLog, test } from './fixtures';

const IPHONE = {
  viewport: { width: 390, height: 664 },
  userAgent:
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1',
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
};

const PROFILES: [string, Record<string, unknown>][] = [
  ['pixel', {}],
  ['iphone', IPHONE],
];

const PREVIEWS = ['start', 'permission', 'watching', 'hint', 'provisional', 'answer', 'answer-left', 'abstain', 'grip', 'unsupported', 'denied', 'inapp'];

/** P(right) as shown: ">99" -> 99.5, "<1" -> 0.5 */
async function shownPercent(page: Page): Promise<number> {
  const raw = (await page.getByTestId('pm-percent').first().innerText()).replace(/\s/g, '');
  if (raw.startsWith('>')) return 99.5;
  if (raw.startsWith('<')) return 0.5;
  return Number(raw.replace('%', ''));
}

const FIXTURES = 'src/predict/__fixtures__';
type Vec = [number, number, number] | null;
type FixtureEvent = [number, Vec, Vec, Vec];

interface NodeReplay {
  driver: string;
  status: string;
  final: boolean;
  pRight: number;
  percentRight: number;
  nUsed: number;
  compare: Record<string, { pRight: number; percentRight: number; k: number }>;
}

/** the page's engine replayed in Node on the same samples (scripts/predictReplay.ts, run with the local tsx) */
function nodeReplay(file: string, driver: string): NodeReplay {
  const out = execFileSync(process.execPath, ['node_modules/tsx/dist/cli.mjs', 'scripts/predictReplay.ts', `${FIXTURES}/${file}`, driver], {
    encoding: 'utf8',
  });
  return JSON.parse(out.trim()) as NodeReplay;
}

/** a committed synthetic fixture's raw samples: [t from 0 (ms), acc, accg, rr as [alpha, beta, gamma]] */
function fixtureEvents(file: string): { engine: string; events: FixtureEvent[] } {
  const fx = JSON.parse(gunzipSync(readFileSync(`${FIXTURES}/${file}`)).toString('utf8')) as {
    engine: string;
    motion: Record<string, (number | null)[]>;
  };
  const m = fx.motion;
  const v = (a: string, b: string, c: string, i: number): Vec =>
    m[a][i] === null || m[b][i] === null || m[c][i] === null ? null : [m[a][i]!, m[b][i]!, m[c][i]!];
  const events = m.t.map(
    (t, i): FixtureEvent => [t!, v('acc_x', 'acc_y', 'acc_z', i), v('accg_x', 'accg_y', 'accg_z', i), v('rr_alpha', 'rr_beta', 'rr_gamma', i)],
  );
  return { engine: fx.engine, events };
}

/**
 * Makes window.__motionListening resolve once the page adds its devicemotion listener (core/sensors.ts does so when
 * the sensor runs). Injected events then start at once, with no Playwright round trip in between, so the source's
 * 1.5 s no-data check cannot fire first on a loaded machine.
 */
async function hookMotionListener(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const w = window as unknown as { __motionListening: Promise<void> };
    let resolve: () => void = () => undefined;
    w.__motionListening = new Promise<void>((r) => (resolve = r));
    const add = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (this: EventTarget, type: string, ...rest: unknown[]) {
      (add as (...a: unknown[]) => void).call(this, type, ...rest);
      if (type === 'devicemotion') setTimeout(resolve, 0);
    } as typeof EventTarget.prototype.addEventListener;
  });
}

/**
 * Arms (before Start) the dispatch of devicemotion events whose handler time is exactly the recorded one: once the page
 * listens, every event is dispatched synchronously with performance.now() (read first thing in core/sensors.ts) pinned
 * to (start + t), then the clock is restored.
 */
async function armExactInjection(page: Page, events: FixtureEvent[]): Promise<void> {
  await page.evaluate((evs) => {
    void (window as unknown as { __motionListening: Promise<void> }).__motionListening.then(() => {
      const perf = performance as Performance & { now: () => number };
      const base = perf.now() + 20;
      let fake = base;
      Object.defineProperty(perf, 'now', { configurable: true, value: () => fake });
      const xyz = (q: number[] | null) => (q ? { x: q[0], y: q[1], z: q[2] } : undefined);
      try {
        for (const [t, acc, accg, rr] of evs as [number, number[] | null, number[] | null, number[] | null][]) {
          fake = base + t;
          window.dispatchEvent(
            new DeviceMotionEvent('devicemotion', {
              acceleration: xyz(acc),
              accelerationIncludingGravity: xyz(accg),
              rotationRate: rr ? { alpha: rr[0], beta: rr[1], gamma: rr[2] } : undefined,
              interval: 16,
            }),
          );
        }
      } finally {
        delete (perf as { now?: unknown }).now; // Performance.prototype.now again
      }
    });
  }, events);
}

/** iOS-style permission API, and webkitCompassHeading (the page then takes the WebKit accelerometer sign) */
async function asIos(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const w = window as unknown as Record<string, unknown>;
    (w.DeviceMotionEvent as { requestPermission?: () => Promise<string> }).requestPermission = () => Promise.resolve('granted');
    Object.defineProperty((w.DeviceOrientationEvent as { prototype: object }).prototype, 'webkitCompassHeading', {
      configurable: true,
      get: () => null,
    });
  });
}

async function noHorizontalScroll(page: Page): Promise<void> {
  const [scroll, inner] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(scroll, 'horizontal scroll').toBeLessThanOrEqual(inner);
}

for (const [profile, use] of PROFILES) {
  test.describe(`predict mode on ${profile}`, () => {
    test.use(use);

    for (const hand of ['right', 'left'] as const) {
      test(`demo: a simulated ${hand} hand gives the known answer, offline and silent`, async ({ page }) => {
        const requests: string[] = [];
        page.on('request', (r) => requests.push(r.url()));
        await page.goto(`/?predict=1&sim=1&model=m4&hand=${hand}&seed=7`);
        await expect(page.getByTestId('pm-start')).toBeVisible();
        await expect(page.locator('#app')).not.toContainText('Loading'); // main.tsx's placeholder is gone once mounted
        await expect(page.getByTestId('pm-demo-badge')).toHaveText(`Demo: simulated ${hand.toUpperCase()} hand`);
        await page.getByTestId('pm-start-btn').click();
        await expect(page.getByTestId('pm-watching')).toBeVisible();
        await expect(page.getByTestId('pm-watching')).toContainText('A simulated person will touch their nose. Watch.');

        const answer = page.getByTestId('pm-answer');
        await expect(answer).toBeVisible({ timeout: 90_000 });
        await expect(answer).toContainText(/P\(right\) = [<>]?\d+ %/);
        await expect(answer).toContainText('probability that the RIGHT hand was used');
        const p = await shownPercent(page);
        if (hand === 'right') expect(p).toBeGreaterThan(50);
        else expect(p).toBeLessThan(50);
        await expect(page.getByTestId('pm-verdict')).toContainText(new RegExp(`${hand.toUpperCase()} hand|Leaning ${hand.toUpperCase()}`));
        // the CNN does not apply to simulated motion: its compare row has no number and no check mark
        await page.getByTestId('pm-compare-toggle').click();
        await expect(page.getByTestId('pm-compare-cnn')).toHaveAttribute('data-na', '1');
        await expect(page.getByTestId('pm-compare-cnn')).not.toContainText('✓');
        await expect(page.getByTestId('pm-compare-cnn')).toContainText('n/a (sim)');
        await page.getByTestId('pm-compare-toggle').click();
        await expect(page.getByTestId('pm-bar')).toHaveAttribute('aria-label', /^LEFT [<>]?\d+ %, RIGHT [<>]?\d+ %$/);
        await expect(page.getByTestId('pm-footer')).toContainText('demo');

        // nothing left the page: no study config, no collector storage, no sound
        expect(requests.filter((u) => u.includes('config.json'))).toEqual([]);
        expect(requests.filter((u) => !u.startsWith('http://127.0.0.1:5173/'))).toEqual([]);
        const dbs = await page.evaluate(async () => (indexedDB.databases ? (await indexedDB.databases()).map((d) => d.name ?? '') : []));
        expect(dbs.filter((n) => n.includes('nosetouch'))).toEqual([]);
        expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
        await noHorizontalScroll(page);
      });
    }

    test('answer: compare every model, switch the model from a row and from the selector, then Again', async ({ page }) => {
      await page.goto('/?predict=1&sim=1&model=m4&hand=right&seed=7');
      await page.getByTestId('pm-start-btn').click();
      await expect(page.getByTestId('pm-answer')).toBeVisible({ timeout: 90_000 });

      await page.getByTestId('pm-compare-toggle').click();
      const table = page.getByTestId('pm-compare');
      await expect(table).toBeVisible();
      for (const id of ['cnn', 'm4', 'phys6', 'yaw']) await expect(page.getByTestId(`pm-compare-${id}`)).toContainText(/\d+ %|n\/a/);
      await expect(page.getByTestId('pm-compare-m4')).toHaveAttribute('aria-pressed', 'true');
      await noHorizontalScroll(page);

      await page.getByTestId('pm-compare-phys6').click();
      await expect(page.getByTestId('pm-model-chip')).toContainText('PHYS6');
      await expect(page.getByTestId('pm-footer')).toContainText(/^phys6 · /);
      await expect(page.getByTestId('pm-compare-phys6')).toHaveAttribute('aria-pressed', 'true');
      expect(new URL(page.url()).searchParams.get('model')).toBe('phys6');

      await page.getByTestId('pm-model-chip').click();
      const sheet = page.getByTestId('pm-model-sheet');
      await expect(sheet).toBeVisible();
      await expect(sheet.getByRole('radio')).toHaveCount(4);
      await expect(sheet).toContainText('Final CNN');
      await expect(sheet).toContainText('Physics logistic');
      await page.getByTestId('pm-model-yaw').click();
      await expect(sheet).toBeHidden();
      await expect(page.getByTestId('pm-model-chip')).toContainText('Yaw');
      expect(new URL(page.url()).searchParams.get('model')).toBe('yaw');

      await page.getByTestId('pm-info-btn').click();
      await expect(page.getByTestId('pm-info-sheet')).toContainText('Nothing leaves this phone.');
      await page.getByTestId('pm-info-sheet').getByRole('button').click();

      // Again: a new script, watching from zero attempts; the model stays
      if (await page.getByTestId('pm-again').isVisible()) await page.getByTestId('pm-again').click();
      else await page.getByTestId('pm-start-over').click();
      await expect(page.getByTestId('pm-watching')).toBeVisible();
      await expect(page.getByTestId('pm-dots')).toHaveAttribute('data-k', '0');
      await expect(page.getByTestId('pm-model-chip')).toContainText('Yaw');
    });

    test('real-sensor path: injected DeviceMotion events with the WebKit sign, after an iOS-style permission', async ({ page }) => {
      await page.addInitScript(() => {
        const w = window as unknown as Record<string, unknown>;
        const DME = w.DeviceMotionEvent as { requestPermission?: () => Promise<string> };
        DME.requestPermission = () => Promise.resolve('granted');
        // every iOS browser has webkitCompassHeading: the page then takes the WebKit (negated) accelerometer sign
        const DOE = w.DeviceOrientationEvent as { prototype: object };
        Object.defineProperty(DOE.prototype, 'webkitCompassHeading', { configurable: true, get: () => null });
      });
      const script = makeDemoScript(7, 'right'); // an iOS script: its acc and accg carry the WebKit sign
      expect(script.engine).toBe('webkit');
      const t0 = script.samples[0].t;
      const events = script.samples.map((s: MotionSample) => [s.t - t0, s.acc, s.accg, s.rr ? [s.rr.alpha, s.rr.beta, s.rr.gamma] : null]);
      await hookMotionListener(page);
      await page.goto('/?predict=1&model=m4');
      await expect(page.getByTestId('pm-start')).toBeVisible();
      await page.evaluate((evs) => {
        // armed before Start: begins when the page listens. Paced in real time (a 16 ms timer), but each event's
        // handler time (performance.now(), read first thing in core/sensors.ts) is pinned to the script's own sample
        // time, as a phone's sensor clock would give it: a loaded test machine that delays the timer then bunches
        // events in time only, never in their timestamps
        const perf = performance as Performance & { now: () => number };
        void (window as unknown as { __motionListening: Promise<void> }).__motionListening.then(() => {
          const start = perf.now();
          let i = 0;
          const vec = (v: number[] | null) => (v ? { x: v[0], y: v[1], z: v[2] } : undefined);
          const timer = setInterval(() => {
            const now = perf.now() - start;
            while (i < evs.length && (evs[i][0] as number) <= now) {
              const [t, acc, accg, rr] = evs[i] as [number, number[] | null, number[] | null, number[] | null];
              Object.defineProperty(perf, 'now', { configurable: true, value: () => start + t });
              try {
                window.dispatchEvent(
                  new DeviceMotionEvent('devicemotion', {
                    acceleration: vec(acc),
                    accelerationIncludingGravity: vec(accg),
                    rotationRate: rr ? { alpha: rr[0], beta: rr[1], gamma: rr[2] } : undefined,
                    interval: 16,
                  }),
                );
              } finally {
                delete (perf as { now?: unknown }).now;
              }
              i += 1;
            }
            if (i >= evs.length) clearInterval(timer);
          }, 16);
        });
      }, events);
      await page.getByTestId('pm-start-btn').click();
      await expect(page.getByTestId('pm-watching')).toBeVisible();
      await expect(page.getByTestId('pm-answer')).toBeVisible({ timeout: 90_000 });
      expect(await shownPercent(page)).toBeGreaterThan(50);
      await expect(page.getByTestId('pm-verdict')).toContainText('RIGHT hand');
      expect(await silenceLog(page)).toEqual({ spoken: [], audioContexts: 0 });
    });

    // the same P(right) as Node: a committed synthetic fixture (Android/Chromium on the Pixel, iOS/WebKit sign on the
    // iPhone profile), its samples injected with their exact times; the default model there, lane m4's here
    const [file, driver] = profile === 'pixel' ? ['syn01_android_right_3touch.json.gz', 'cnn'] : ['syn02_ios_left_gap_repeat.json.gz', 'm4'];
    test(`injected fixture ${file.split('_')[0]}: the page shows the P(right) of the same engine in Node (${driver}, then every model)`, async ({ page }) => {
      const { engine, events } = fixtureEvents(file);
      const node = nodeReplay(file, driver);
      expect(node.final).toBe(true);
      if (engine === 'webkit') await asIos(page);
      await hookMotionListener(page);
      await page.goto(driver === 'cnn' ? '/?predict=1' : `/?predict=1&model=${driver}`);
      await expect(page.locator('.pm')).toHaveAttribute('data-engine', 'real');
      await armExactInjection(page, events);
      await page.getByTestId('pm-start-btn').click();

      const answer = page.getByTestId('pm-answer');
      await expect(answer).toBeVisible({ timeout: 30_000 });
      await expect(answer).toContainText('probability that the RIGHT hand was used');
      const shown = Number(await page.getByTestId('pm-prob').getAttribute('data-p'));
      expect(Math.abs(shown - node.pRight), `page ${shown} vs Node ${node.pRight}`).toBeLessThanOrEqual(1e-12);
      const norm = (x: string): string => x.replace(/\s/g, ' ');
      expect(norm(await page.getByTestId('pm-percent').innerText())).toBe(norm(formatPercent(node.percentRight)));
      await expect(page.getByTestId('pm-verdict')).toContainText(`${node.pRight > 0.5 ? 'RIGHT' : 'LEFT'} hand`);
      await expect(page.getByTestId('pm-footer')).toContainText(new RegExp(`^${driver} · [0-9a-f]{8} · v`));

      // compare: every model's own aggregation after the same attempts, as in Node
      await page.getByTestId('pm-compare-toggle').click();
      let worst = Math.abs(shown - node.pRight);
      for (const id of ['cnn', 'm4', 'phys6', 'yaw']) {
        const p = Number(await page.getByTestId(`pm-compare-${id}`).getAttribute('data-p'));
        worst = Math.max(worst, Math.abs(p - node.compare[id].pRight));
        expect(Math.abs(p - node.compare[id].pRight), `compare ${id}: page ${p} vs Node ${node.compare[id].pRight}`).toBeLessThanOrEqual(1e-12);
      }
      console.log(`${file}: page vs Node, driver ${driver} P(right) ${shown}, worst |dP| over the driver and 4 compare rows ${worst}`);

      // Again: a new stream with the same sensors, zero attempts, the model kept
      await page.getByTestId('pm-again').click();
      await expect(page.getByTestId('pm-watching')).toBeVisible();
      await expect(page.getByTestId('pm-dots')).toHaveAttribute('data-k', '0');
      await expect(page.getByTestId('pm-footer')).toContainText(new RegExp(`^${driver} · `));
    });

    test('permission: the prompt screen while it is open, the denied message after a refusal', async ({ page, context }) => {
      await context.addInitScript(() => {
        const DME = (window as unknown as { DeviceMotionEvent: { requestPermission?: () => Promise<string> } }).DeviceMotionEvent;
        DME.requestPermission = () =>
          new URLSearchParams(location.search).get('deny') === '1' ? Promise.resolve('denied') : new Promise<string>(() => undefined);
      });
      await page.goto('/?predict=1');
      await page.getByTestId('pm-start-btn').click();
      await expect(page.getByTestId('pm-permission')).toContainText('Allow motion access');

      await page.goto('/?predict=1&deny=1');
      await page.getByTestId('pm-start-btn').click();
      const unsupported = page.getByTestId('pm-unsupported');
      await expect(unsupported).toHaveAttribute('data-failure', 'permission_denied');
      await expect(unsupported).toContainText('Motion access was refused.');
      await expect(page.getByTestId(/^pm-action-(reload|copy)$/)).toBeVisible(); // iOS: copy the link for after quitting Safari
    });

    test('no sensor data: the unsupported screen after 1.5 s, and the demo link leads to &sim=1', async ({ page }) => {
      await page.goto('/?predict=1&lang=en');
      await page.getByTestId('pm-start-btn').click();
      const unsupported = page.getByTestId('pm-unsupported');
      await expect(unsupported).toBeVisible({ timeout: 10_000 });
      await expect(unsupported).toHaveAttribute('data-failure', /no_events|permission_denied|not_supported/);
      await page.getByTestId(/pm-action-demo/).first().click();
      await expect(page).toHaveURL(/[?&]sim=1/);
      await expect(page.getByTestId('pm-start')).toBeVisible();
      await expect(page.getByTestId('pm-demo-badge')).toBeVisible();
    });
  });
}

test('the desktop demo without &model drives with lane m4; the default route still renders the collector', async ({ page }) => {
  await page.goto('/?predict=1&sim=1&hand=left&seed=3');
  await expect(page.locator('.pm')).toHaveAttribute('data-engine', 'real');
  await expect(page.getByTestId('pm-footer')).toContainText(/^m4 · [0-9a-f]{8} · v.* · demo$/);
  await page.goto('/?predict=1');
  await expect(page.getByTestId('pm-footer')).toContainText(/^cnn · [0-9a-f]{8} · v/);
  await page.goto('/');
  await expect(page.locator('#app')).not.toBeEmpty();
  await expect(page.locator('.pm')).toHaveCount(0);
});

test.describe('predict mode layout', () => {
  for (const [w, h] of [
    [360, 640],
    [640, 360],
  ] as const) {
    test(`every state fits ${w}x${h} without horizontal scroll`, async ({ page }) => {
      await page.setViewportSize({ width: w, height: h });
      for (const name of PREVIEWS) {
        for (const lang of ['en', 'he']) {
          await page.goto(`/?predict=1&preview=${name}&lang=${lang}`);
          await expect(page.locator('.pm')).toHaveAttribute('data-screen', /\w+/);
          await noHorizontalScroll(page);
        }
      }
      await page.goto('/?predict=1&preview=answer&view=compare&debug=1');
      await expect(page.getByTestId('pm-compare')).toBeVisible();
      await expect(page.getByTestId('pm-debug')).toBeVisible();
      await noHorizontalScroll(page);
    });
  }

  test('Hebrew: right to left, the P line isolated, the hand bar still LEFT on the left', async ({ page }) => {
    await page.goto('/?predict=1&preview=answer&lang=he');
    await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
    await expect(page.locator('html')).toHaveAttribute('lang', 'he');
    await expect(page.getByTestId('pm-answer')).toContainText('ההסתברות שהשתמשו ביד ימין');
    await expect(page.getByTestId('pm-answer')).toContainText(/P\(right\) = \d+ %/);
    const words = page.locator('.pm-bar-word');
    const left = await words.nth(0).boundingBox();
    const right = await words.nth(1).boundingBox();
    expect(await words.nth(0).innerText()).toBe('שמאל');
    expect(left!.x).toBeLessThan(right!.x);
    await expect(page.getByTestId('pm-verdict')).toContainText('יד ימין');
  });

  test('the screens each carry their test id', async ({ page }) => {
    const expected: Record<string, string> = {
      start: 'pm-start',
      permission: 'pm-permission',
      watching: 'pm-watching',
      provisional: 'pm-watching',
      answer: 'pm-answer',
      abstain: 'pm-abstain',
      grip: 'pm-grip',
      unsupported: 'pm-unsupported',
    };
    for (const [name, id] of Object.entries(expected)) {
      await page.goto(`/?predict=1&preview=${name}`);
      await expect(page.getByTestId(id)).toBeVisible();
    }
    await page.goto('/?predict=1&preview=grip');
    await expect(page.getByTestId('pm-grip')).toContainText('Turn the screen towards your face');
    await page.goto('/?predict=1&preview=abstain');
    await expect(page.getByTestId('pm-abstain')).toContainText('No touch found');
  });
});
