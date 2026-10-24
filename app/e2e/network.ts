/**
 * The pure half of the e2e network guard (see e2e/fixtures.ts): no Playwright import here, so the unit
 * test tests/e2eHarness.test.ts can check it.
 */

/** Hosts a test may talk to: the dev server only (127.0.0.1 / localhost / the baseURL's host). */
export function localHosts(baseURL: string | undefined): Set<string> {
  const hosts = new Set(['127.0.0.1', 'localhost', '[::1]']);
  if (baseURL) {
    try {
      hosts.add(new URL(baseURL).hostname);
    } catch {
      // not a URL: the defaults only
    }
  }
  return hosts;
}

/** The study config the tests run with: the committed config.json, with uploads switched off. */
export function offlineConfig(raw: unknown): unknown {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return raw;
  const cfg = raw as Record<string, unknown>;
  const upload = cfg.upload && typeof cfg.upload === 'object' ? (cfg.upload as Record<string, unknown>) : {};
  return { ...cfg, upload: { ...upload, mode: 'none' } };
}

/** The guard's route matcher: requests to another host, and the study config (rewritten). */
export function guardedUrl(url: URL, local: Set<string>): boolean {
  return !local.has(url.hostname) || url.pathname.endsWith('/config.json');
}
