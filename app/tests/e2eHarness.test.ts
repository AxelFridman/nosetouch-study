/**
 * The e2e harness must keep the researcher's computer silent (spec v2 decision 8). CI does not run the
 * Playwright suite, so this unit test checks its set-up instead:
 *  - every spec takes `test`/`expect` from e2e/fixtures.ts (which stubs speech and Web Audio in every
 *    page), and no e2e file imports a value from '@playwright/test' directly (types only);
 *  - both Playwright configs launch Chromium with --mute-audio and have an action timeout;
 *  - nothing opts into sound (?audio=1) or into a visible browser;
 *  - the in-page stub itself: speech is recorded, never spoken; an AudioContext cannot be created.
 */
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import { silenceAudio } from '../e2e/silence';

const APP = fileURLToPath(new URL('../', import.meta.url));
const E2E = fileURLToPath(new URL('../e2e/', import.meta.url));
const CONFIGS = ['playwright.config.ts', 'playwright.live.config.ts'];
const read = (path: string): string => readFileSync(path, 'utf8');
const e2eFiles = readdirSync(E2E).filter((f) => f.endsWith('.ts'));
const specs = e2eFiles.filter((f) => f.endsWith('.spec.ts'));

/** Names imported as VALUES from `module` (type-only imports and `type X` specifiers excluded). */
function valueImports(src: string, module: string): string[] {
  const names: string[] = [];
  const escaped = module.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
  const re = new RegExp(String.raw`import\s+(type\s+)?([^'";]*?)\s+from\s*['"]${escaped}['"]`, 'g');
  for (const m of src.matchAll(re)) {
    if (m[1]) continue; // import type { ... }
    const clause = m[2].trim();
    const braces = /\{([^}]*)\}/.exec(clause);
    const outside = clause.replace(/\{[^}]*\}/, '').replace(/,/g, ' ').trim();
    if (outside) names.push(...outside.split(/\s+/).filter((n) => n && n !== 'as')); // default / namespace
    if (braces) {
      for (const spec of braces[1].split(',').map((s) => s.trim()).filter(Boolean)) {
        if (!spec.startsWith('type ')) names.push(spec.split(/\s+as\s+/)[0]);
      }
    }
  }
  return names;
}

describe('e2e harness: silence', () => {
  it('finds the e2e specs', () => {
    expect(specs.length).toBeGreaterThanOrEqual(6);
    expect(e2eFiles).toContain('fixtures.ts');
  });

  it.each(specs)('%s takes test and expect from ./fixtures', (file) => {
    const src = read(E2E + file);
    const fromFixtures = valueImports(src, './fixtures');
    expect(fromFixtures, `${file} must import { test, expect } from './fixtures'`).toEqual(expect.arrayContaining(['test', 'expect']));
  });

  it.each(e2eFiles.filter((f) => f !== 'fixtures.ts'))('%s imports no value from @playwright/test (types only)', (file) => {
    expect(valueImports(read(E2E + file), '@playwright/test')).toEqual([]);
  });

  it('the import check itself tells values from types', () => {
    const src = [
      "import { expect, type Page, test as t } from '@playwright/test';",
      "import type { Locator } from '@playwright/test';",
      "import pw, * as all from '@playwright/test';",
    ].join('\n');
    expect(valueImports(src, '@playwright/test')).toEqual(['expect', 'test', 'pw', '*', 'all']);
    expect(valueImports("import { test, expect, silenceLog } from './fixtures';", './fixtures')).toEqual(['test', 'expect', 'silenceLog']);
  });

  it('fixtures.ts installs the stub as an init script of every context, before the app runs', () => {
    const src = read(`${E2E}fixtures.ts`);
    expect(src).toMatch(/context\.addInitScript\(\s*silenceAudio\s*\)/);
    expect(src).toMatch(/base\.extend/);
  });

  it.each(CONFIGS)('%s launches Chromium with --mute-audio and has an action timeout', (file) => {
    const src = read(APP + file);
    expect(src).toContain("'--mute-audio'");
    expect(src).toMatch(/launchOptions:\s*\{\s*args:\s*\[[^\]]*'--mute-audio'/);
    expect(src).toMatch(/actionTimeout:\s*15_000/);
  });

  it('no e2e file or config opts into sound or into a visible browser', () => {
    for (const path of [...e2eFiles.map((f) => E2E + f), ...CONFIGS.map((f) => APP + f)]) {
      const src = read(path);
      expect(src, path).not.toMatch(/[?&]audio=1/);
      expect(src, path).not.toMatch(/headless:\s*false/);
      expect(src, path).not.toMatch(/--headed|headed:\s*true|slowMo/);
    }
  });
});

describe('e2e harness: the in-page stub (silenceAudio)', () => {
  const g = globalThis as unknown as Record<string, unknown>;
  afterEach(() => {
    delete g.window;
  });

  function install(): Record<string, unknown> {
    const fakeWindow: Record<string, unknown> = {
      speechSynthesis: { speak: () => { throw new Error('the real speechSynthesis must never be reached'); } },
      AudioContext: class {},
      webkitAudioContext: class {},
    };
    g.window = fakeWindow;
    silenceAudio();
    return fakeWindow;
  }

  it('records speech instead of speaking it', () => {
    const w = install();
    const synth = w.speechSynthesis as { speak(u: { text: string }): void; getVoices(): unknown[]; cancel(): void };
    synth.speak({ text: 'Left hand. Fast' });
    synth.cancel();
    expect(synth.getVoices()).toEqual([]);
    expect(w.__nosetouchSilence).toEqual({ spoken: ['Left hand. Fast'], audioContexts: 0 });
  });

  it('makes AudioContext and webkitAudioContext unconstructible, and counts the attempts', () => {
    const w = install();
    for (const key of ['AudioContext', 'webkitAudioContext']) {
      const Ctor = w[key] as new () => unknown;
      expect(() => new Ctor()).toThrow(/audio is disabled/);
    }
    expect((w.__nosetouchSilence as { audioContexts: number }).audioContexts).toBe(2);
  });
});
