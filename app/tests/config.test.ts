import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  CONFIG_CACHE_KEY,
  CONFIG_TIMEOUT_MS,
  effectiveUpload,
  fetchStudyConfig,
  isValidEndpoint,
  loadStudyConfig,
  redactProfile,
  redactSession,
  refetchConfigWhenOnline,
  sanitizeStudyConfig,
  type LoadedConfig,
} from '../src/core/config';
import { DEFAULT_STUDY_CONFIG } from '../src/core/constants';
import type { StudyConfig } from '../src/core/types';
import { makeProfile, makeSession } from './helpers/dataFixtures';
import { expectValid } from './helpers/schema';

const okJson = (body: unknown) => vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('sanitizeStudyConfig', () => {
  it('returns a fresh copy of the defaults for anything that is not an object', () => {
    for (const raw of [null, undefined, 42, 'x', [], [1, 2]]) {
      const cfg = sanitizeStudyConfig(raw);
      expect(cfg).toEqual(DEFAULT_STUDY_CONFIG);
      expect(cfg).not.toBe(DEFAULT_STUDY_CONFIG);
      expect(cfg.upload).not.toBe(DEFAULT_STUDY_CONFIG.upload);
    }
  });

  it('deep-merges a partial config over the defaults', () => {
    const cfg = sanitizeStudyConfig({
      study_code: 'pilot-2026',
      upload: { mode: 'apps_script', endpoint: ' https://script.google.com/macros/s/AKfy/exec ', study_key: ' key-1 ' },
      contact: { email: 'lab@example.org' },
      unknown_field: true,
    });
    expect(cfg).toEqual<StudyConfig>({
      ...DEFAULT_STUDY_CONFIG,
      study_code: 'pilot-2026',
      upload: { mode: 'apps_script', endpoint: 'https://script.google.com/macros/s/AKfy/exec', study_key: 'key-1' },
      contact: { name: '', email: 'lab@example.org' },
    });
    expect(cfg).not.toHaveProperty('unknown_field');
  });

  it('falls back field by field when values are invalid', () => {
    const cfg = sanitizeStudyConfig({
      study_code: 'bad code!',
      upload: { mode: 'ftp', endpoint: 'http://insecure.example.org/ingest', study_key: 7 },
      upload_nickname: 'yes',
      consent_version: '   ',
      contact: 'nobody',
      default_language: 'fr',
      default_preset: 'mega',
      min_age: '18',
    });
    expect(cfg).toEqual({ ...DEFAULT_STUDY_CONFIG, upload: { mode: 'none', endpoint: '', study_key: '' } });
  });

  it('validates study_code like the backend id pattern', () => {
    expect(sanitizeStudyConfig({ study_code: 'ok_Code-1' }).study_code).toBe('ok_Code-1');
    for (const bad of ['', '../x', 'a b', 'x'.repeat(65), 'שלום', 12]) {
      expect(sanitizeStudyConfig({ study_code: bad }).study_code).toBe(DEFAULT_STUDY_CONFIG.study_code);
    }
  });

  it('keeps valid scalars and clamps min_age to an integer within 0..120', () => {
    const cfg = sanitizeStudyConfig({
      upload_nickname: true,
      consent_version: '2027-01-v2',
      default_language: 'he',
      default_preset: 'full',
      min_age: 17.6,
    });
    expect(cfg).toMatchObject({ upload_nickname: true, consent_version: '2027-01-v2', default_language: 'he', default_preset: 'full', min_age: 18 });
    expect(sanitizeStudyConfig({ min_age: 16 }).min_age).toBe(16);
    expect(sanitizeStudyConfig({ min_age: 121 }).min_age).toBe(18);
    expect(sanitizeStudyConfig({ min_age: -1 }).min_age).toBe(18);
    expect(sanitizeStudyConfig({ min_age: Number.NaN }).min_age).toBe(18);
  });

  it('the shipped public/config.json is valid as is', () => {
    const raw = JSON.parse(readFileSync(new URL('../public/config.json', import.meta.url), 'utf8'));
    expect(sanitizeStudyConfig(raw)).toEqual(raw);
  });
});

describe('isValidEndpoint', () => {
  it('accepts https URLs and same-origin relative paths only', () => {
    for (const ok of [
      'https://script.google.com/macros/s/AKfy/exec',
      'https://abc.trycloudflare.com/ingest',
      '/api/ingest',
      './ingest',
      '../api/ingest',
    ]) {
      expect(isValidEndpoint(ok), ok).toBe(true);
    }
    for (const bad of [
      '',
      'http://example.org/ingest',
      '//evil.example.org/ingest',
      '/\\evil.example.org',
      'javascript:alert(1)',
      'ftp://example.org',
      'https://user:pass@example.org/x',
      'https://exa mple.org',
      'api/ingest',
      'https://',
    ]) {
      expect(isValidEndpoint(bad), bad).toBe(false);
    }
  });
});

describe('loadStudyConfig', () => {
  it("fetches './config.json' without the HTTP cache and sanitises it", async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const f = okJson({ study_code: 'lab-a', upload: { mode: 'json_post', endpoint: '/api/ingest', study_key: 'k' } });
    const cfg = await loadStudyConfig(f);
    expect(f).toHaveBeenCalledWith('./config.json', expect.objectContaining({ cache: 'no-store' }));
    expect(cfg.study_code).toBe('lab-a');
    expect(cfg.upload).toEqual({ mode: 'json_post', endpoint: '/api/ingest', study_key: 'k' });
    expect(console.warn).not.toHaveBeenCalled();
  });

  it('never throws: network errors, HTTP errors and broken JSON all give the defaults', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const failing = [
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      }),
      vi.fn(async () => new Response('not found', { status: 404 })),
      vi.fn(async () => new Response('{"study_code": ', { status: 200 })),
    ];
    for (const f of failing) expect(await loadStudyConfig(f)).toEqual(DEFAULT_STUDY_CONFIG);
    expect(console.warn).toHaveBeenCalledTimes(3);
  });

  it('gives up after the timeout instead of blocking boot', async () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    let aborted = false;
    const f = vi.fn(
      (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>(() => {
          init?.signal?.addEventListener('abort', () => (aborted = true));
        }),
    );
    const p = loadStudyConfig(f);
    await vi.advanceTimersByTimeAsync(CONFIG_TIMEOUT_MS);
    expect(await p).toEqual(DEFAULT_STUDY_CONFIG);
    expect(aborted).toBe(true);
  });

  it('a relative URL without a document (Node) still resolves to the defaults', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(await loadStudyConfig()).toEqual(DEFAULT_STUDY_CONFIG);
  });
});

/** A Storage stand-in; `broken` makes every access throw (storage blocked). */
function memoryStorage(init: Record<string, string> = {}, broken = false) {
  const map = new Map(Object.entries(init));
  const check = () => {
    if (broken) throw new DOMException('The operation is insecure.', 'SecurityError');
  };
  return {
    map,
    getItem: (k: string) => {
      check();
      return map.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      check();
      map.set(k, v);
    },
  };
}

const offline = () =>
  vi.fn(async () => {
    throw new TypeError('Failed to fetch');
  });

describe('fetchStudyConfig: the last good config.json survives an offline start', () => {
  const deployed = { study_code: 'wave-2', consent_version: '2027-02-v3', upload: { mode: 'apps_script', endpoint: 'https://script.google.com/macros/s/AKfy/exec', study_key: 'k2' } };

  it('remembers every config that loaded, and says it came from the network', async () => {
    const storage = memoryStorage();
    const loaded = await fetchStudyConfig(okJson(deployed), storage);
    expect(loaded.source).toBe('network');
    expect(loaded.config.study_code).toBe('wave-2');
    const saved = JSON.parse(storage.map.get(CONFIG_CACHE_KEY) ?? 'null');
    expect(saved.config).toEqual(loaded.config);
    expect(typeof saved.url).toBe('string');
  });

  it('uses the remembered copy, not the defaults, whenever loading fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const storage = memoryStorage();
    const good = (await fetchStudyConfig(okJson(deployed), storage)).config;
    const failures = [
      offline(),
      vi.fn(async () => new Response('busy', { status: 503 })),
      vi.fn(async () => new Response('<html>Wi-Fi login</html>', { status: 200 })), // captive portal
      vi.fn(async () => new Response('[1, 2]', { status: 200 })), // JSON, but not a config object
    ];
    for (const f of failures) {
      const loaded = await fetchStudyConfig(f, storage);
      expect(loaded).toEqual({ config: good, source: 'cache' });
      expect(loaded.config.upload.mode).toBe('apps_script'); // uploads stay on
    }
    expect(await loadStudyConfig(offline(), storage)).toEqual(good);
  });

  it('also after a timeout', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const storage = memoryStorage();
    await fetchStudyConfig(okJson(deployed), storage);
    vi.useFakeTimers();
    const p = fetchStudyConfig(vi.fn(() => new Promise<Response>(() => undefined)), storage);
    await vi.advanceTimersByTimeAsync(CONFIG_TIMEOUT_MS);
    expect((await p).source).toBe('cache');
  });

  it('falls back to the defaults without a usable copy: none, corrupt, another deployment, storage blocked', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const otherApp = JSON.stringify({ url: 'https://lab.github.io/other-study/config.json', config: deployed });
    for (const storage of [
      memoryStorage(),
      memoryStorage({ [CONFIG_CACHE_KEY]: '{"url": ' }),
      memoryStorage({ [CONFIG_CACHE_KEY]: otherApp }),
      memoryStorage({}, true),
      null,
    ]) {
      expect(await fetchStudyConfig(offline(), storage)).toEqual({ config: DEFAULT_STUDY_CONFIG, source: 'defaults' });
    }
    // a blocked storage never breaks a successful load either
    expect((await fetchStudyConfig(okJson(deployed), memoryStorage({}, true))).source).toBe('network');
  });

  it('sanitises the remembered copy again (it is plain localStorage text)', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const storage = memoryStorage();
    await fetchStudyConfig(okJson(deployed), storage);
    const entry = JSON.parse(storage.map.get(CONFIG_CACHE_KEY)!);
    entry.config.upload.endpoint = 'http://insecure.example.org';
    storage.map.set(CONFIG_CACHE_KEY, JSON.stringify(entry));
    expect((await fetchStudyConfig(offline(), storage)).config.upload.endpoint).toBe('');
  });
});

describe('refetchConfigWhenOnline', () => {
  const flush = () => new Promise((r) => setTimeout(r, 0));
  const fresh = sanitizeStudyConfig({ study_code: 'wave-2' });

  it("loads config.json at each 'online' event until the network answers, applies it once, then stops", async () => {
    const target = new EventTarget();
    const results: LoadedConfig[] = [
      { config: DEFAULT_STUDY_CONFIG, source: 'defaults' },
      { config: fresh, source: 'network' },
    ];
    const load = vi.fn(async () => results.shift() ?? { config: fresh, source: 'network' as const });
    const apply = vi.fn();
    refetchConfigWhenOnline(target, apply, load);
    target.dispatchEvent(new Event('online'));
    await flush();
    expect(apply).not.toHaveBeenCalled(); // still offline in effect: nothing replaced
    target.dispatchEvent(new Event('online'));
    await flush();
    expect(apply).toHaveBeenCalledExactlyOnceWith(fresh);
    target.dispatchEvent(new Event('online'));
    await flush();
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('runs one load at a time and can be stopped', async () => {
    const target = new EventTarget();
    let release: (v: LoadedConfig) => void = () => undefined;
    const load = vi.fn(() => new Promise<LoadedConfig>((r) => (release = r)));
    const apply = vi.fn();
    const stop = refetchConfigWhenOnline(target, apply, load);
    target.dispatchEvent(new Event('online'));
    target.dispatchEvent(new Event('online'));
    expect(load).toHaveBeenCalledTimes(1);
    stop();
    release({ config: fresh, source: 'network' });
    await flush();
    expect(apply).not.toHaveBeenCalled();
  });
});

// ───────────────────────────── service worker (public/sw.js) ─────────────────────────────

const SCOPE = 'https://lab.github.io/nosetouch/';
const INDEX_HTML = '<!doctype html><script type="module" src="./assets/index-abc123.js"></script><link rel="stylesheet" href="./assets/index-def456.css">';

/**
 * Runs public/sw.js in a sandbox with in-memory caches and a scripted network. The caches match like
 * the Cache API: by URL, and, unless ignoreVary, only when the headers named in the stored
 * response's Vary are equal on both requests.
 */
function loadServiceWorker(files: Record<string, string>, opts: { vary?: string } = {}) {
  const net = { offline: false, failing: new Set<string>() };
  const listeners: Record<string, (e: unknown) => void> = {};
  type Entry = { req: Request; res: Response };
  const stores = new Map<string, Map<string, Entry>>();
  const keyOf = (r: RequestInfo | URL) => (typeof r === 'string' ? r : r instanceof URL ? r.href : r.url);
  const asRequest = (r: RequestInfo | URL) => (r instanceof Request ? r : new Request(keyOf(r)));
  const cacheFor = (m: Map<string, Entry>) => ({
    match: async (r: RequestInfo, o?: { ignoreVary?: boolean }) => {
      const req = asRequest(r);
      const hit = m.get(req.url);
      if (!hit) return undefined;
      const vary = (hit.res.headers.get('vary') ?? '').split(',').map((h) => h.trim()).filter(Boolean);
      if (!o?.ignoreVary && vary.some((h) => h === '*' || hit.req.headers.get(h) !== req.headers.get(h))) return undefined;
      return hit.res.clone();
    },
    put: async (r: RequestInfo, res: Response) => void m.set(keyOf(r), { req: asRequest(r), res }),
    keys: async () => [...m.values()].map((e) => e.req),
    delete: async (r: RequestInfo) => m.delete(keyOf(r)),
  });
  const caches = {
    open: async (name: string) => {
      if (!stores.has(name)) stores.set(name, new Map());
      return cacheFor(stores.get(name)!);
    },
    keys: async () => [...stores.keys()],
    delete: async (name: string) => stores.delete(name),
  };
  // a same-origin answer, as a worker sees it (Node's Response is of type 'default')
  const basic = (res: Response) => Object.defineProperty(res, 'type', { value: 'basic' });
  const headers: Record<string, string> = opts.vary ? { Vary: opts.vary } : {};
  const fetchImpl = async (input: RequestInfo | URL) => {
    const url = keyOf(input);
    if (net.offline || net.failing.has(url)) throw new TypeError('Failed to fetch');
    const path = url.split('?')[0];
    return path in files
      ? basic(new Response(files[path], { status: 200, headers }))
      : basic(new Response('not found', { status: 404, headers }));
  };
  const self = {
    registration: { scope: SCOPE },
    location: new URL(SCOPE),
    addEventListener: (type: string, fn: (e: unknown) => void) => void (listeners[type] = fn),
    skipWaiting: () => undefined,
    clients: { claim: async () => undefined },
  };
  runInNewContext(readFileSync(new URL('../public/sw.js', import.meta.url), 'utf8'), {
    self,
    caches,
    fetch: fetchImpl,
    Request,
    Response,
    URL,
    setTimeout,
    clearTimeout,
    console,
  });
  const install = async () => {
    let done: Promise<unknown> = Promise.resolve();
    listeners.install({ waitUntil: (p: Promise<unknown>) => (done = p) });
    await done;
  };
  const get = async (url: string, requestHeaders: Record<string, string> = {}): Promise<Response> => {
    let answer: Promise<Response> | undefined;
    listeners.fetch({
      request: new Request(url, { headers: requestHeaders }),
      respondWith: (p: Promise<Response>) => (answer = p),
      waitUntil: (p: Promise<unknown>) => void p.catch(() => undefined),
    });
    if (!answer) throw new Error(`the worker did not answer ${url}`);
    return answer;
  };
  const cached = () => new Set([...stores.values()].flatMap((m) => [...m.keys()]));
  return { net, install, get, cached };
}

const SITE: Record<string, string> = {
  [SCOPE]: INDEX_HTML,
  [`${SCOPE}index.html`]: INDEX_HTML,
  [`${SCOPE}manifest.webmanifest`]: '{}',
  [`${SCOPE}assets/index-abc123.js`]: 'console.log(1)',
  [`${SCOPE}assets/index-def456.css`]: 'body{}',
  [`${SCOPE}config.json`]: JSON.stringify({ study_code: 'wave-2', upload: { mode: 'apps_script', endpoint: 'https://x.example/exec', study_key: 'k' } }),
};

describe('service worker: config.json is part of the offline shell', () => {
  it('pre-caches config.json under origin + pathname at install, so the first offline start has it', async () => {
    const sw = loadServiceWorker(SITE);
    await sw.install();
    expect(sw.cached()).toContain(`${SCOPE}config.json`);
    sw.net.offline = true;
    // the app asks for './config.json' (no-store); any query string maps to the same entry
    const res = await sw.get(`${SCOPE}config.json?v=2`);
    expect(JSON.parse(await res.text()).study_code).toBe('wave-2');
  });

  it('installs without config.json (missing or unreachable): it is optional', async () => {
    const { [`${SCOPE}config.json`]: _omit, ...withoutConfig } = SITE;
    const missing = loadServiceWorker(withoutConfig);
    await missing.install();
    expect(missing.cached()).toContain(`${SCOPE}index.html`);
    expect(missing.cached()).not.toContain(`${SCOPE}config.json`);

    const unreachable = loadServiceWorker(SITE);
    unreachable.net.failing.add(`${SCOPE}config.json`);
    await unreachable.install();
    expect(unreachable.cached()).toContain(`${SCOPE}assets/index-abc123.js`);
    expect(unreachable.cached()).not.toContain(`${SCOPE}config.json`);
  });

  it('starts offline even where the server answers with "Vary: Origin" (vite preview)', async () => {
    const sw = loadServiceWorker(SITE, { vary: 'Origin' });
    await sw.install();
    sw.net.offline = true;
    // a module script request carries an Origin header; the pre-cached copy was fetched without one
    const js = await sw.get(`${SCOPE}assets/index-abc123.js`, { Origin: 'https://lab.github.io' });
    expect(await js.text()).toBe('console.log(1)');
    const config = await sw.get(`${SCOPE}config.json`);
    expect(JSON.parse(await config.text()).study_code).toBe('wave-2');
  });

  it('a missing shell file still fails the install (unchanged)', async () => {
    const { [`${SCOPE}manifest.webmanifest`]: _omit, ...broken } = SITE;
    await expect(loadServiceWorker(broken).install()).rejects.toThrow(/precache: HTTP 404/);
  });
});

describe('effectiveUpload', () => {
  const cfg = sanitizeStudyConfig({
    upload: { mode: 'apps_script', endpoint: 'https://script.google.com/macros/s/AKfy/exec', study_key: 'study-key' },
  });

  it('returns a copy of config.upload without an override', () => {
    const up = effectiveUpload(cfg, null);
    expect(up).toEqual(cfg.upload);
    expect(up).not.toBe(cfg.upload);
  });

  it('overlays non-empty, valid override fields one by one', () => {
    expect(effectiveUpload(cfg, { endpoint: '/api/ingest', mode: 'json_post' })).toEqual({
      mode: 'json_post',
      endpoint: '/api/ingest',
      study_key: 'study-key',
    });
    expect(effectiveUpload(cfg, { study_key: ' other ' }).study_key).toBe('other');
    expect(effectiveUpload(cfg, { mode: 'none' }).mode).toBe('none');
  });

  it('ignores empty or invalid override fields (a half-filled form cannot break uploads)', () => {
    expect(effectiveUpload(cfg, { endpoint: '', study_key: '', mode: undefined })).toEqual(cfg.upload);
    expect(effectiveUpload(cfg, { endpoint: 'http://insecure.example.org' })).toEqual(cfg.upload);
    expect(effectiveUpload(cfg, { mode: 'carrier-pigeon' as never })).toEqual(cfg.upload);
    expect(effectiveUpload(cfg, {})).toEqual(cfg.upload);
  });
});

describe('redaction', () => {
  it('redactProfile drops the nickname unless upload_nickname is set, without mutating the input', () => {
    const profile = makeProfile({ nickname: 'Dana' });
    const hidden = redactProfile(profile, { ...DEFAULT_STUDY_CONFIG, upload_nickname: false });
    expect(hidden.nickname).toBeNull();
    expect({ ...hidden, nickname: 'Dana' }).toEqual(profile);
    expect(profile.nickname).toBe('Dana');
    expect(hidden.ehi_sf).not.toBe(profile.ehi_sf);

    const kept = redactProfile(profile, { ...DEFAULT_STUDY_CONFIG, upload_nickname: true });
    expect(kept.nickname).toBe('Dana');
    expect(kept).not.toBe(profile);
  });

  it('redactSession redacts the participant snapshot and keeps the record schema-valid', () => {
    const session = makeSession();
    const redacted = redactSession(session, DEFAULT_STUDY_CONFIG);
    expect(redacted.participant.nickname).toBeNull();
    expect(session.participant.nickname).toBe('Dana');
    expect({ ...redacted, participant: session.participant }).toEqual(session);
    expectValid('SessionRecord', redacted);
  });
});
