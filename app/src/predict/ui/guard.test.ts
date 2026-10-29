/**
 * Predict mode's isolation from the collector (docs/stage3/APP_SPEC.md sections 1.4 and 2.1):
 *  - no predict source imports the collector's storage, uploader, config fetch, cue sink (sound), trial
 *    controller, online_v1 segmentation, calibration, protocol, exporter or anything under app/;
 *  - main.tsx loads the predict chunk with a dynamic import only, before rendering the collector;
 *  - mount() blocks network connections with a CSP meta (connect-src 'none').
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { CSP_CONTENT, installNoNetworkPolicy } from '../mount';

const PREDICT = fileURLToPath(new URL('../', import.meta.url));
const SRC = fileURLToPath(new URL('../../', import.meta.url));

function sources(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...sources(p));
    else if (/\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

const FORBIDDEN = [
  /core\/storage['"]/,
  /core\/uploader['"]/,
  /core\/config['"]/,
  /core\/trialController['"]/,
  /core\/segmentation['"]/,
  /core\/calibration['"]/,
  /core\/protocol['"]/,
  /core\/exporter['"]/,
  /\bcreateCueSink\b/,
  /from\s+['"](\.\.\/)+app\//,
  /import\(\s*['"](\.\.\/)+app\//,
];

describe('predict mode isolation', () => {
  const files = sources(PREDICT);

  it('finds the predict sources', () => {
    expect(files.some((f) => f.endsWith('PredictApp.tsx'))).toBe(true);
    expect(files.some((f) => f.endsWith('mount.tsx'))).toBe(true);
  });

  it.each(files.map((f) => [f.slice(PREDICT.length).replace(/\\/g, '/'), f]))('%s imports nothing of the collector that stores, uploads, fetches or sounds', (_name, file) => {
    const src = readFileSync(file, 'utf8');
    const imports = src.split('\n').filter((l) => /^\s*import\b|\bimport\(|^\s*export .* from /.test(l) || /createCueSink/.test(l));
    for (const re of FORBIDDEN) expect(imports.join('\n'), `${re}`).not.toMatch(re);
  });

  it('no predict source uses localStorage, IndexedDB, fetch, XHR, sendBeacon, WebSocket or audio', () => {
    for (const file of files) {
      const src = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '');
      expect(src, file).not.toMatch(/\blocalStorage\b|\bindexedDB\b|\bfetch\(|XMLHttpRequest|sendBeacon|WebSocket|AudioContext|speechSynthesis/);
    }
  });

  it('main.tsx branches before the collector and loads the predict chunk dynamically', () => {
    const main = readFileSync(join(SRC, 'main.tsx'), 'utf8');
    expect(main).toMatch(/import\('\.\/predict\/mount'\)/);
    expect(main).not.toMatch(/from\s+['"]\.\/predict/);
    const branch = main.indexOf("import('./predict/mount')");
    const collector = main.indexOf('render(<App />');
    expect(branch).toBeGreaterThan(0);
    expect(collector).toBeGreaterThan(branch); // the collector renders in the else branch only
    expect(main.indexOf('serviceWorker.register')).toBeGreaterThan(collector);
  });
});

describe('mount: no network for the life of the page', () => {
  function fakeDocument() {
    const children: { httpEquiv: string; content: string; attrs: Record<string, string> }[] = [];
    const doc = {
      head: {
        querySelector: () => children.find((c) => c.attrs['data-predict'] === '1') ?? null,
        appendChild: (el: (typeof children)[number]) => children.push(el),
      },
      createElement: () => {
        const el = { httpEquiv: '', content: '', attrs: {} as Record<string, string>, setAttribute: (k: string, v: string) => (el.attrs[k] = v) };
        return el;
      },
    };
    return { doc: doc as unknown as Document, children };
  }

  it("adds one connect-src 'none' policy (idempotent)", () => {
    const { doc, children } = fakeDocument();
    installNoNetworkPolicy(doc);
    installNoNetworkPolicy(doc);
    expect(children).toHaveLength(1);
    expect(children[0].httpEquiv).toBe('Content-Security-Policy');
    expect(children[0].content).toBe(CSP_CONTENT);
    expect(CSP_CONTENT).toBe("connect-src 'none'");
  });
});
