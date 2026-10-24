/**
 * Renders the gesture illustration (src/ui/GestureAnimation.tsx) into example images for the user, the
 * docs and presentations, in docs/img/gesture/:
 *   right_upright_natural.gif, left_upright_natural.gif, right_angled_natural.gif,
 *   left_top_left_natural.gif (sideways grip), right_upright_fast.gif, right_upright_slow.gif,
 *   either_hand_natural.gif (participant's choice: the right hand, then the left),
 *   right_upright_natural_{start,halfway,touch}.png (key frames, 2x) and
 *   how_to_touch_your_nose.png (the three steps, side view and "what you see", with captions).
 *
 * How: the dev gallery (gesture-gallery.html) shows one gesture (?only=<id>&bare=1); every CSS
 * animation is frozen at time t (window.gxSetTime = negative animation-delay + paused) and the item is
 * screenshotted in headless Chromium (muted, never makes a sound). Frames are captured at 2x (a 440-px
 * item gives 880-px frames); scripts/make_gesture_gif.py (Pillow) builds one shared palette that keeps
 * the drawing's token colours exactly (the touch burst must survive: the script checks it) and writes
 * looping GIFs (and the captioned sheet). Frame rates follow the speed: 30 fps for fast (its approach
 * lasts 0.35 s), 20 fps for natural, 15 fps for slow and for the two-hand loop.
 *
 *   npx tsx scripts/render_gesture_images.ts [--out ../docs/img/gesture] [--fps <all GIFs>] [--width 440]
 *       [--scale 1] [--only right_upright_fast,...] [--port 5299] [--python <exe>] [--work <dir>]
 *       [--keep-frames]
 *
 * Reuses a dev server already answering on --port; otherwise starts
 * `npx vite --port 5299 --strictPort --host 127.0.0.1` and stops it (the whole process tree) at the end,
 * also on errors and Ctrl+C. Python: --python, $PYTHON, ../analysis/.venv, py -3, python3, python (the
 * first one that can import PIL).
 */
import { chromium, type Browser, type Page } from '@playwright/test';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const APP_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GALLERY = 'gesture-gallery.html';

interface GifSpec {
  name: string;
  /** gallery item id: hand.orientation.speed.contact.variant.view */
  id: string;
  /** frames per second (--fps overrides) */
  fps: number;
}

const GIFS: GifSpec[] = [
  { name: 'right_upright_natural', id: 'right.upright_toward.natural.straight.standard.both', fps: 20 },
  { name: 'left_upright_natural', id: 'left.upright_toward.natural.straight.standard.both', fps: 20 },
  { name: 'right_angled_natural', id: 'right.upright_toward.natural.angled.standard.both', fps: 20 },
  { name: 'left_top_left_natural', id: 'left.top_left_toward.natural.straight.standard.both', fps: 20 },
  { name: 'right_upright_fast', id: 'right.upright_toward.fast.straight.standard.both', fps: 30 },
  { name: 'right_upright_slow', id: 'right.upright_toward.slow.straight.standard.both', fps: 15 },
  { name: 'either_hand_natural', id: 'any.upright_toward.natural.straight.standard.both', fps: 15 },
];
/** the touch burst of the light theme (gesture.css --gx-burst): every GIF must show it */
const BURST = '#e08600';
const KEY_ID = 'right.upright_toward.natural.straight.standard.both';
const SHEET_CAPTIONS = ['1  Arm straight out in front, phone in your hand', '2  Bring the phone in', '3  Touch your nose with it (any part of the phone)'];

// ───────────────────────────── arguments ─────────────────────────────

interface Args {
  out: string;
  /** null = each GIF's own rate */
  fps: number | null;
  width: number;
  /** GIF size relative to the 2x frames (1 = 2 × width pixels wide) */
  scale: number;
  port: number;
  only: string[] | null;
  python: string | null;
  work: string | null;
  keepFrames: boolean;
}

function parseArgs(argv: string[]): Args {
  const a: Args = {
    out: path.resolve(APP_DIR, '..', 'docs', 'img', 'gesture'),
    fps: null,
    width: 440,
    scale: 1,
    port: 5299,
    only: null,
    python: null,
    work: null,
    keepFrames: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const v = () => {
      const x = argv[++i];
      if (x === undefined) throw new Error(`${k} needs a value`);
      return x;
    };
    if (k === '--out') a.out = path.resolve(v());
    else if (k === '--fps') a.fps = Number(v());
    else if (k === '--width') a.width = Number(v());
    else if (k === '--scale') a.scale = Number(v());
    else if (k === '--port') a.port = Number(v());
    else if (k === '--only') a.only = v().split(',').map((s) => s.trim()).filter(Boolean);
    else if (k === '--python') a.python = v();
    else if (k === '--work') a.work = path.resolve(v());
    else if (k === '--keep-frames') a.keepFrames = true;
    else if (k === '--help' || k === '-h') {
      console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('*/')[0]);
      process.exit(0);
    } else throw new Error(`unknown argument ${k}`);
  }
  if (a.fps !== null && !(a.fps > 0 && a.fps <= 50)) throw new Error('--fps must be in (0, 50]');
  if (!(a.scale > 0 && a.scale <= 1)) throw new Error('--scale must be in (0, 1]');
  if (!(a.width >= 200 && a.width <= 1200)) throw new Error('--width must be 200..1200');
  return a;
}

// ───────────────────────────── dev server ─────────────────────────────

let server: ChildProcess | null = null;

async function answers(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return r.ok;
  } catch {
    return false;
  }
}

function stopServer(): void {
  const child = server;
  server = null;
  if (!child || child.pid === undefined || child.exitCode !== null) return;
  if (process.platform === 'win32') {
    // npx runs through cmd.exe: kill the whole tree (cmd → node → vite/esbuild)
    spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
  } else {
    try {
      process.kill(-child.pid, 'SIGTERM'); // the process group created by detached: true
    } catch {
      child.kill('SIGTERM');
    }
  }
}

async function ensureServer(port: number): Promise<string> {
  const base = `http://127.0.0.1:${port}`;
  if (await answers(`${base}/${GALLERY}`)) {
    console.log(`using the dev server already running on ${base}`);
    return base;
  }
  console.log(`starting: npx vite --port ${port} --strictPort --host 127.0.0.1`);
  server = spawn(`npx vite --port ${port} --strictPort --host 127.0.0.1`, {
    cwd: APP_DIR,
    shell: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
    windowsHide: true,
  });
  let log = '';
  server.stdout?.on('data', (d) => (log += String(d)));
  server.stderr?.on('data', (d) => (log += String(d)));
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`vite exited (${server.exitCode}):\n${log}`);
    if (await answers(`${base}/${GALLERY}`)) return base;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`vite did not answer on ${base} within 60 s:\n${log}`);
}

// ───────────────────────────── python ─────────────────────────────

function findPython(explicit: string | null): string[] {
  const venv =
    process.platform === 'win32'
      ? path.resolve(APP_DIR, '..', 'analysis', '.venv', 'Scripts', 'python.exe')
      : path.resolve(APP_DIR, '..', 'analysis', '.venv', 'bin', 'python');
  const candidates: string[][] = [];
  if (explicit) candidates.push([explicit]);
  if (process.env.PYTHON) candidates.push([process.env.PYTHON]);
  if (existsSync(venv)) candidates.push([venv]);
  candidates.push(['py', '-3'], ['python3'], ['python']);
  for (const [cmd, ...pre] of candidates) {
    const r = spawnSync(cmd, [...pre, '-c', 'import PIL'], { stdio: 'ignore', windowsHide: true });
    if (r.status === 0) return [cmd, ...pre];
  }
  throw new Error('no Python with Pillow found (pip install pillow, or pass --python)');
}

function runPython(py: string[], args: string[]): void {
  const [cmd, ...pre] = py;
  const r = spawnSync(cmd, [...pre, path.join(APP_DIR, 'scripts', 'make_gesture_gif.py'), ...args], {
    stdio: 'inherit',
    windowsHide: true,
  });
  if (r.status !== 0) throw new Error(`make_gesture_gif.py ${args[0]} failed (${r.status})`);
}

// ───────────────────────────── capture ─────────────────────────────

interface GestureInfo {
  key: string;
  cycle: number;
  takes: number;
  mid: number;
  touch: number;
  touches: number[];
}

async function openItem(page: Page, base: string, query: string): Promise<GestureInfo> {
  await page.goto(`${base}/${GALLERY}?${query}`);
  await page.waitForSelector('body[data-gx-ready="1"]');
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  const info = await page.evaluate(() => (window as unknown as { gxInfo(): GestureInfo[] }).gxInfo()[0]);
  if (!info || !(info.cycle > 0)) throw new Error(`no gesture rendered for ?${query}`);
  return info;
}

async function freezeAt(page: Page, t: number): Promise<void> {
  await page.evaluate((time) => (window as unknown as { gxSetTime(t: number): void }).gxSetTime(time), t);
  // two animation frames: the new delay is applied and painted
  await page.evaluate(() => new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r()))));
}

async function captureGif(page: Page, base: string, spec: GifSpec, fps: number, a: Args, dir: string): Promise<void> {
  const info = await openItem(page, base, `only=${spec.id}&bare=1&theme=light&paused=1&t=0&size=${a.width}`);
  const n = Math.round(info.cycle * fps);
  // a loop that alternates hands fades out and in around the cut at t = 0: start the GIF a little later,
  // so its first frame (the one shown where GIFs do not play) is the full start pose
  const t0 = info.takes > 1 ? 0.2 : 0;
  mkdirSync(dir, { recursive: true });
  const shot = page.locator('.gal-shot').first();
  for (let i = 0; i < n; i++) {
    await freezeAt(page, t0 + i / fps);
    await shot.screenshot({ path: path.join(dir, `frame_${String(i).padStart(3, '0')}.png`) });
  }
  // the freeze must actually move the drawing: the start and the touch must differ
  const iTouch = Math.min(n - 1, Math.round((info.touch + 0.12 - t0) * fps));
  const f0 = readFileSync(path.join(dir, 'frame_000.png'));
  const fT = readFileSync(path.join(dir, `frame_${String(iTouch).padStart(3, '0')}.png`));
  if (f0.equals(fT)) throw new Error(`${spec.name}: frames do not change (animation freeze not applied?)`);
  console.log(`${spec.name}: ${n} frames at ${fps} fps (${info.cycle} s loop, ${info.takes} take(s))`);
}

async function main(): Promise<void> {
  const a = parseArgs(process.argv.slice(2));
  const py = findPython(a.python);
  const work = a.work ?? mkdtempSync(path.join(tmpdir(), 'nosetouch-gesture-'));
  mkdirSync(work, { recursive: true });
  mkdirSync(a.out, { recursive: true });
  const wanted = (name: string) => !a.only || a.only.includes(name);

  let browser: Browser | null = null;
  try {
    const base = await ensureServer(a.port);
    browser = await chromium.launch({ args: ['--mute-audio'] });
    const context = await browser.newContext({
      viewport: { width: a.width + 160, height: 900 },
      deviceScaleFactor: 2,
      colorScheme: 'light',
      reducedMotion: 'no-preference',
    });
    const page = await context.newPage();
    page.on('pageerror', (e) => console.error('page error:', e.message));

    for (const spec of GIFS) {
      if (!wanted(spec.name)) continue;
      const dir = path.join(work, spec.name);
      rmSync(dir, { recursive: true, force: true });
      const fps = a.fps ?? spec.fps;
      await captureGif(page, base, spec, fps, a, dir);
      runPython(py, [
        'gif',
        '--frames',
        dir,
        '--out',
        path.join(a.out, `${spec.name}.gif`),
        '--fps',
        String(fps),
        '--scale',
        String(a.scale),
        '--require',
        BURST,
      ]);
    }

    if (wanted('keyframes')) {
      const info = await openItem(page, base, `only=${KEY_ID}&bare=1&theme=light&paused=1&t=0&size=${a.width}`);
      const shot = page.locator('.gal-shot').first();
      const moments: [string, number][] = [
        ['start', 0.15],
        ['halfway', info.mid],
        ['touch', info.touch + 0.12],
      ];
      for (const [label, t] of moments) {
        await freezeAt(page, t);
        const raw = path.join(work, `key_${label}.png`);
        await shot.screenshot({ path: raw });
        runPython(py, ['png', '--src', raw, '--out', path.join(a.out, `right_upright_natural_${label}.png`)]);
      }
    }

    if (wanted('sheet')) {
      // the component's own reduced-motion strip: 1 → 2 → 3, side view and "what you see"
      await openItem(page, base, `only=${KEY_ID}&bare=1&theme=light&motion=reduce&size=720`);
      const shot = page.locator('.gal-shot').first();
      const box = await shot.boundingBox();
      if (!box) throw new Error('sheet: item not visible');
      const centers = await page.$$eval('.gx-frow-side .gx-frame', (els) =>
        els.map((e) => {
          const r = e.getBoundingClientRect();
          return r.left + r.width / 2;
        }),
      );
      if (centers.length !== 3) throw new Error(`sheet: expected 3 frames, found ${centers.length}`);
      const strip = path.join(work, 'sheet_strip.png');
      await shot.screenshot({ path: strip });
      const px = centers.map((x) => Math.round((x - box.x) * 2)).join(',');
      runPython(py, [
        'sheet',
        '--strip',
        strip,
        '--out',
        path.join(a.out, 'how_to_touch_your_nose.png'),
        '--centers',
        px,
        '--captions',
        SHEET_CAPTIONS.join('|'),
        '--title',
        'How to touch your nose with the phone',
        '--note',
        'Top: seen from the side. Bottom: what you see (your right hand; the left hand is mirrored).',
      ]);
    }
    await context.close();
  } finally {
    if (browser) await browser.close();
    stopServer();
    if (!a.keepFrames && !a.work) rmSync(work, { recursive: true, force: true });
  }
  console.log(`images in ${a.out}`);
}

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    stopServer();
    process.exit(130);
  });
}
process.on('exit', () => stopServer());

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  stopServer();
  process.exit(1);
});
