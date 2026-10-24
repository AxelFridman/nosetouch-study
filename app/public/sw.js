/*
 * NoseTouch service worker: offline app shell for the collector PWA. Hand-written and copied to dist/
 * unchanged (no build step, no Workbox). src/main.tsx registers it in production builds only. Its
 * scope is the folder it is served from, so the same file works at https://<user>.github.io/<repo>/
 * and under `npm run preview`.
 *
 * What it handles (everything else goes to the network untouched):
 *  - navigations           network first, so every deploy is picked up; the cached index.html when
 *                          the network fails or stalls for NAV_TIMEOUT_MS. The network request
 *                          revalidates the browser's HTTP cache (cache: 'no-cache'; a cheap 304 when
 *                          nothing changed): GitHub Pages serves index.html with max-age=600, and a
 *                          page opened within 10 minutes of the previous visit would otherwise get the
 *                          old index.html from the HTTP cache after a deploy
 *  - ./assets/*            cache first: Vite puts a content hash in these file names, so a cached
 *                          copy can never be stale (matched ignoring Vary, see cacheFirst)
 *  - ./config.json         network first (study settings change without a rebuild); the last copy
 *                          when offline
 *  - ./manifest.webmanifest network first, cached copy offline
 * What it NEVER touches: non-GET requests and cross-origin requests. Uploads (POST to
 * script.google.com, or to /api/ingest for the local receiver) must reach the network exactly as
 * the page sent them; a worker in the middle could only add failure modes.
 *
 * Install pre-caches ./, ./index.html and ./manifest.webmanifest, plus the hashed assets that
 * index.html references: the first visit loaded its assets before this worker existed, and without
 * them the app could not start offline. For the same reason a fresh index.html (after a deploy)
 * replaces the cached one only once all of its assets are cached too. ./config.json is pre-cached
 * as well (under origin + pathname, the key the fetch handler uses): the first visit fetched it
 * before this worker existed, and without a copy the first offline start would run on the built-in
 * defaults (wrong study_code and consent_version, no institution/region/contact, uploads off). It is
 * optional: a deployment without config.json still installs, since the app then uses its defaults
 * anyway. skipWaiting + clients.claim make a new worker take over at once; that is safe because
 * navigations are network first and older hashed assets stay cached.
 * CACHE_VERSION: bump it when this file's caching scheme changes, or when a release must never start
 * from an older cached shell (activate then deletes the old caches; the browser installs a changed
 * sw.js at the next visit: register() checks the worker script without the HTTP cache). Other app
 * releases need no bump.
 *  - 1: app 0.1.x.
 *  - 2: app 0.2.0 (the "street" visit): navigations revalidate the HTTP cache, and the v0.1 shell is
 *       dropped, so a phone that starts offline or on a stalled network after the v0.2.0 deploy never
 *       falls back to the v0.1 flow (calibration, long profile).
 */
'use strict';

const CACHE_VERSION = 2;
const CACHE_PREFIX = 'nosetouch-';
const SCOPE = self.registration.scope; // absolute URL ending in '/'
const SCOPE_PATH = new URL(SCOPE).pathname;
// The scope is part of the name: all GitHub Pages project sites of one user share the origin
// https://<user>.github.io (and so its caches), and each copy of the app must clean up only its own.
const CACHE = CACHE_PREFIX + 'v' + CACHE_VERSION + ':' + SCOPE;
const INDEX_URL = new URL('index.html', SCOPE).href;
const SHELL = [SCOPE, INDEX_URL, new URL('manifest.webmanifest', SCOPE).href];
// pre-cached when the server has it; the key has no query string, like the fetch handler's
const CONFIG_KEY = new URL('config.json', SCOPE).href;
const NAV_TIMEOUT_MS = 5000;
const CONFIG_TIMEOUT_MS = 4000; // below the app's own 8 s config timeout, so the cached copy still arrives in time
const MAX_ASSET_ENTRIES = 60; // old builds' hashed files pile up across deploys; keep the newest ones
// Asset URLs in Vite's index.html: src="./assets/index-<hash>.js", href="./assets/index-<hash>.css", ...
const ASSET_REF_RE = /(?:src|href)\s*=\s*["']?((?:\.\/)?assets\/[^"'\s>]+)/g;

const noop = () => undefined;

/** Path relative to the scope ('' for the app root), or null outside the scope. */
function appPath(url) {
  return url.pathname.startsWith(SCOPE_PATH) ? url.pathname.slice(SCOPE_PATH.length) : null;
}

/**
 * Chrome refuses to serve a response that went through a redirect to a navigation request, so a
 * redirected shell response is stored as a fresh copy without the redirect flag.
 */
function storable(res) {
  if (!res.redirected) return Promise.resolve(res);
  return res.blob().then(
    (body) => new Response(body, { status: res.status, statusText: res.statusText, headers: res.headers }),
  );
}

function isFresh200(res) {
  return !!res && res.status === 200 && res.type === 'basic';
}

/**
 * Caches the hashed assets an index.html references (those not cached yet). false when one cannot be
 * fetched: the caller then keeps the previous index.html, because an index.html whose assets are
 * missing could not start offline.
 */
async function cacheAssets(cache, html) {
  const urls = new Set();
  for (const m of html.matchAll(ASSET_REF_RE)) urls.add(new URL(m[1], SCOPE).href);
  for (const url of urls) {
    if (await cache.match(url)) continue;
    let res;
    try {
      res = await fetch(url);
    } catch (e) {
      return false;
    }
    if (!isFresh200(res)) return false;
    await cache.put(url, res);
  }
  return true;
}

/** Stores a fresh copy of the app's page, but only once everything it needs is cached. */
async function storeIndex(res) {
  const cache = await caches.open(CACHE);
  if (await cacheAssets(cache, await res.clone().text())) await cache.put(INDEX_URL, await storable(res));
}

/**
 * The navigation request with cache: 'no-cache': the browser revalidates its HTTP cache with the
 * server (If-None-Match / If-Modified-Since) instead of reusing a copy that is still "fresh" by
 * max-age, so a deploy shows at the next start. Mode and redirect handling stay those of the original
 * (a Request built from a navigation becomes same-origin, which a same-origin page is). The original
 * request when a browser refuses to build it.
 */
function revalidating(req) {
  try {
    return new Request(req, { cache: 'no-cache' });
  } catch (e) {
    return req;
  }
}

// ───────────────────────────── lifecycle ─────────────────────────────

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(precache());
});

async function precache() {
  const cache = await caches.open(CACHE);
  // cache: 'reload' skips the HTTP cache, so a deploy is never pre-cached from a stale copy.
  // Any failure fails the install (config.json excepted, see the top), and the browser retries at
  // the next visit; the app works online meanwhile. The assets go in first, so a half-done install
  // never leaves a shell without them.
  const shell = [];
  for (const url of SHELL) {
    const res = await fetch(new Request(url, { cache: 'reload' }));
    if (!res.ok) throw new Error('precache: HTTP ' + res.status + ' for ' + url);
    shell.push([url, await storable(res)]);
  }
  let config = null;
  try {
    config = await fetch(new Request(CONFIG_KEY, { cache: 'reload' }));
  } catch (e) {
    config = null; // offline after the shell arrived: the fetch handler caches it on the next start
  }
  if (isFresh200(config)) shell.push([CONFIG_KEY, await storable(config)]);
  const index = shell.find(([url]) => url === INDEX_URL)[1];
  if (!(await cacheAssets(cache, await index.clone().text()))) throw new Error('precache: an asset is missing');
  for (const [url, res] of shell) await cache.put(url, res);
}

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      const mine = (n) => n.startsWith(CACHE_PREFIX) && n.endsWith(':' + SCOPE);
      await Promise.all(names.filter((n) => mine(n) && n !== CACHE).map((n) => caches.delete(n)));
      await self.clients.claim();
    })(),
  );
});

// ───────────────────────────── fetch ─────────────────────────────

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return; // uploads and every other state change: not ours, ever
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // script.google.com, tunnels, CDNs: not ours
  if (req.mode === 'navigate') {
    const path = appPath(url);
    // only the app's own page refreshes the cached copy (a 404 page or another document must not)
    const own = path === '' || path === 'index.html';
    event.respondWith(networkFirst(event, [INDEX_URL, SCOPE], NAV_TIMEOUT_MS, own ? storeIndex : null, revalidating(req)));
    return;
  }
  const path = appPath(url);
  if (path === null) return;
  if (path.startsWith('assets/')) {
    event.respondWith(cacheFirst(event));
  } else if (path === 'config.json' || path === 'manifest.webmanifest') {
    const key = url.origin + url.pathname; // without the query string
    const store = (res) => caches.open(CACHE).then((cache) => cache.put(key, res));
    event.respondWith(networkFirst(event, [key], CONFIG_TIMEOUT_MS, store));
  }
});

async function cacheFirst(event) {
  const cache = await caches.open(CACHE);
  // ignoreVary: a hashed file never changes, but a server that answers with "Vary: Origin" (vite
  // preview does) would otherwise never match the page's module-script request (it carries an Origin
  // header, the pre-cached copy was fetched without one), and no offline start could load the app.
  const hit = await cache.match(event.request, { ignoreVary: true });
  if (hit) return hit;
  const res = await fetch(event.request);
  if (isFresh200(res)) {
    event.waitUntil(
      cache
        .put(event.request, res.clone())
        .then(() => trimAssets(cache))
        .catch(noop),
    );
  }
  return res;
}

/**
 * The network's answer when it arrives within `timeoutMs`; otherwise (or when it fails) the cached
 * copy under the first of `keys` that exists. A fresh 200 answer is handed to `store` (when given),
 * also when it arrives late. Without any cached copy the page simply waits for the network (or gets
 * its error), as it would without a service worker. `request`: what goes to the network (default:
 * the page's own request).
 */
function networkFirst(event, keys, timeoutMs, store, request) {
  const network = fetch(request || event.request);
  // Registered before the page's handler below, so clone() runs before the body is consumed.
  const update = network.then((res) => (store && isFresh200(res) ? store(res.clone()) : undefined));
  event.waitUntil(update.catch(noop));
  const cached = () =>
    caches.open(CACHE).then(async (cache) => {
      for (const key of keys) {
        const hit = await cache.match(key);
        if (hit) return hit;
      }
      return undefined;
    });
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (res) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(res);
    };
    const timer = setTimeout(() => {
      cached().then((hit) => {
        if (hit) finish(hit);
      }, noop);
    }, timeoutMs);
    network.then(finish, (err) => {
      cached().then(
        (hit) => {
          if (hit) finish(hit);
          else if (!done) {
            done = true;
            clearTimeout(timer);
            reject(err);
          }
        },
        () => {
          if (!done) {
            done = true;
            clearTimeout(timer);
            reject(err);
          }
        },
      );
    });
  });
}

async function trimAssets(cache) {
  // cache.keys() lists entries in insertion order, i.e. oldest first
  const assets = (await cache.keys()).filter((r) => {
    const path = appPath(new URL(r.url));
    return path !== null && path.startsWith('assets/');
  });
  for (let i = 0; i < assets.length - MAX_ASSET_ENTRIES; i++) await cache.delete(assets[i]);
}
