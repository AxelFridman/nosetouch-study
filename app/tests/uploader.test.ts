import 'fake-indexeddb/auto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { openLocalStore, type LocalStore } from '../src/core/storage';
import {
  backoffDelayMs,
  buildEnvelope,
  createUploadQueue,
  isDue,
  orderForUpload,
  sendEnvelope,
  uploadEnabled,
  UPLOAD_TIMEOUT_MS,
} from '../src/core/uploader';
import type { OutboxItem, SessionRecord, UploadConfig, UploadEnvelope } from '../src/core/types';
import { isRefused, summarizeOutbox, uploadLabel } from '../src/app/screens/DataScreen';
import { admin as enAdmin } from '../src/i18n/en/admin';
import { admin as heAdmin } from '../src/i18n/he/admin';
import { APPS_SCRIPT_URL, appsScriptCfg, makeEnvelope, makeProfile, makeSession, makeTrial } from './helpers/dataFixtures';
import { expectValid } from './helpers/schema';

// ───────────────────────────── helpers ─────────────────────────────

type FetchArgs = [RequestInfo | URL, RequestInit | undefined];
type Reply = (() => Response) | Error;

const json = (body: unknown, status = 200): (() => Response) => () =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
// 204/205/304 responses must not carry a body (the Response constructor throws otherwise)
const text = (body: string, status = 200): (() => Response) => () =>
  new Response([204, 205, 304].includes(status) ? null : body, { status });
/** Lets pending IndexedDB work (scheduled with the real setImmediate) run to completion. */
const drain = async (): Promise<void> => {
  for (let i = 0; i < 200; i++) await new Promise<void>((r) => setImmediate(r));
};
const networkError = (): Error => new TypeError('Failed to fetch');
/** What Code.gs answers for a wrong study key (the phone still has an old config.json). */
const refusal = (): (() => Response) =>
  json({ ok: false, error: 'study key does not match', code: 'bad_study_key', retry: false, received_at: '2026-10-23T10:00:00.000Z' });

/** A fetch double that replays `replies` in order (the last one repeats) and records each call. */
function fakeFetch(...replies: Reply[]) {
  const queue = [...replies];
  return vi.fn(async (_url: RequestInfo | URL, _init?: RequestInit): Promise<Response> => {
    const r = queue.length > 1 ? queue.shift()! : queue[0];
    if (r instanceof Error) throw r;
    return r();
  });
}

function sentEnvelopes(f: ReturnType<typeof fakeFetch>): UploadEnvelope[] {
  return f.mock.calls.map(([, init]) => JSON.parse(String(init?.body)) as UploadEnvelope);
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

let dbCounter = 0;
async function freshStore(): Promise<LocalStore> {
  dbCounter += 1;
  return openLocalStore(`test-uploader-${dbCounter}`);
}

/** Saves a session with `nTrials` trials and enqueues everything (session first, like sessionFlow). */
async function seedSession(store: LocalStore, sid: string, nTrials: number, pid = 'P-ABCDEF'): Promise<string[]> {
  await store.saveSession(makeSession({ session_id: sid, participant: makeProfile({ participant_id: pid }), participant_id: pid }));
  await store.enqueue('session', sid, sid);
  const trialIds: string[] = [];
  for (let i = 0; i < nTrials; i++) {
    const tid = `${sid}-t${i}`;
    await store.saveTrial(makeTrial({ trial_id: tid, session_id: sid, participant_id: pid, trial_index: i }));
    await store.enqueue('trial', tid, sid);
    trialIds.push(tid);
  }
  return trialIds;
}

const byId = async (store: LocalStore, id: string): Promise<OutboxItem | undefined> =>
  (await store.listOutbox()).find((i) => i.id === id);

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

// ───────────────────────────── envelope / transport ─────────────────────────────

describe('buildEnvelope', () => {
  it('wraps the record, maps an empty study key to null and validates against the schema', () => {
    const env = buildEnvelope('trial', makeTrial(), appsScriptCfg({ study_key: '' }));
    expect(env.kind).toBe('trial');
    expect(env.study_key).toBeNull();
    expect(Number.isFinite(Date.parse(env.sent_at))).toBe(true);
    expectValid('UploadEnvelope', env);
    expect(buildEnvelope('session', makeSession(), appsScriptCfg()).study_key).toBe('k-123');
    expectValid('UploadEnvelope', buildEnvelope('session', makeSession(), appsScriptCfg()));
  });
});

describe('sendEnvelope — apps_script', () => {
  const env = makeEnvelope('trial', makeTrial(), 'k-123');

  it('POSTs the JSON as a CORS simple request (no headers at all) and confirms a readable {ok:true}', async () => {
    const f = fakeFetch(json({ ok: true, kind: 'trial', id: 'trial-0001' }));
    const res = await sendEnvelope(appsScriptCfg(), env, f);
    expect(res).toMatchObject({ ok: true, confirmed: true, error: null });
    expect(f).toHaveBeenCalledTimes(1);
    const [url, init] = f.mock.calls[0] as FetchArgs;
    expect(url).toBe(APPS_SCRIPT_URL);
    expect(init?.method).toBe('POST');
    expect(typeof init?.body).toBe('string'); // string body → text/plain;charset=UTF-8
    expect(JSON.parse(String(init?.body))).toEqual(env);
    expect(init).not.toHaveProperty('headers');
    expect(init?.mode).not.toBe('no-cors');
    expect(init?.credentials).toBe('omit');
    expect(init?.redirect).toBe('follow');
    expect(init?.keepalive).toBeUndefined();
    expect(init?.signal).toBeInstanceOf(AbortSignal);
  });

  it('a TypeError (dead deployment, CORS-less error page, offline) is a transient failure', async () => {
    const res = await sendEnvelope(appsScriptCfg(), env, fakeFetch(networkError()));
    expect(res.ok).toBe(false);
    expect(res.confirmed).toBe(false);
    expect(res.transient).toBe(true);
    expect(res.permanent).toBe(false);
    expect(res.error).toMatch(/network error: Failed to fetch/);
  });

  it('HTTP 500 is a transient failure; HTTP 400 is not', async () => {
    const r500 = await sendEnvelope(appsScriptCfg(), env, fakeFetch(text('oops', 500)));
    expect(r500).toMatchObject({ ok: false, error: 'HTTP 500', transient: true, permanent: false });
    const r400 = await sendEnvelope(appsScriptCfg(), env, fakeFetch(json({ ok: false, error: 'bad' }, 400)));
    expect(r400).toMatchObject({ ok: false, error: 'HTTP 400: bad', transient: false, permanent: true });
  });

  it('{ok:false} without retry:true is the server refusing this record (permanent); retry:true is transient', async () => {
    const rejected = await sendEnvelope(appsScriptCfg(), env, fakeFetch(refusal()));
    expect(rejected).toMatchObject({ ok: false, confirmed: false, transient: false, permanent: true });
    expect(rejected.error).toContain('study key does not match');
    const noRetryField = await sendEnvelope(appsScriptCfg(), env, fakeFetch(json({ ok: false, error: 'bad ids' })));
    expect(noRetryField).toMatchObject({ ok: false, transient: false, permanent: true });
    const busy = await sendEnvelope(appsScriptCfg(), env, fakeFetch(json({ ok: false, error: 'busy', retry: true })));
    expect(busy).toMatchObject({ ok: false, transient: true, permanent: false });
    const ok = await sendEnvelope(appsScriptCfg(), env, fakeFetch(json({ ok: true })));
    expect(ok).toMatchObject({ ok: true, permanent: false });
  });

  it('an unreadable 200 reply is never taken as success, nor as a refusal', async () => {
    const html = await sendEnvelope(appsScriptCfg(), env, fakeFetch(text('<!DOCTYPE html><html>Sign in</html>')));
    expect(html).toMatchObject({ ok: false, transient: true, permanent: false });
    expect(html.error).toMatch(/unexpected response/);
    const noFlag = await sendEnvelope(appsScriptCfg(), env, fakeFetch(json({ stored: true })));
    expect(noFlag).toMatchObject({ ok: false, permanent: false });
  });

  it('times out after 45 s through the AbortController', async () => {
    vi.useFakeTimers();
    let aborted = false;
    const f = vi.fn(
      (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener('abort', () => {
            aborted = true;
            reject(new DOMException('The operation was aborted.', 'AbortError'));
          });
        }),
    );
    const p = sendEnvelope(appsScriptCfg(), env, f);
    await vi.advanceTimersByTimeAsync(UPLOAD_TIMEOUT_MS - 1);
    expect(aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(await p).toMatchObject({ ok: false, error: 'timeout after 45 s', transient: true, permanent: false });
    expect(aborted).toBe(true);
  });

  it('still gives up after 45 s when the fetch implementation ignores the abort signal', async () => {
    vi.useFakeTimers();
    const p = sendEnvelope(appsScriptCfg(), env, vi.fn(() => new Promise<Response>(() => undefined)));
    await vi.advanceTimersByTimeAsync(UPLOAD_TIMEOUT_MS);
    expect(await p).toMatchObject({ ok: false, error: 'timeout after 45 s' });
  });
});

describe('sendEnvelope — json_post and disabled modes', () => {
  const cfg: UploadConfig = { mode: 'json_post', endpoint: '/api/ingest', study_key: '' };
  const env = makeEnvelope('session', makeSession());

  it('sends application/json and accepts {ok:true}, an empty 2xx body or JSON without ok:false', async () => {
    const f = fakeFetch(json({ ok: true }), text('', 204), json({ stored: 1 }));
    for (let i = 0; i < 3; i++) expect((await sendEnvelope(cfg, env, f)).ok).toBe(true);
    const [url, init] = f.mock.calls[0] as FetchArgs;
    expect(url).toBe('/api/ingest');
    expect(init?.headers).toEqual({ 'Content-Type': 'application/json' });
  });

  it('rejects {ok:false} and non-JSON replies', async () => {
    expect((await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'bad ids' })))).error).toContain('bad ids');
    const html = await sendEnvelope(cfg, env, fakeFetch(text('<html>index</html>')));
    expect(html).toMatchObject({ ok: false, transient: true, permanent: false });
  });

  it("the local receiver's answers: a 4xx refusal is permanent, its 500 is transient, a bare 4xx page is neither", async () => {
    const r400 = await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'study key does not match', code: 'bad_study_key', retry: false }, 400)));
    expect(r400).toMatchObject({ ok: false, transient: false, permanent: true, error: 'HTTP 400: study key does not match' });
    const r413 = await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'payload too large', code: 'too_large', retry: false }, 413)));
    expect(r413).toMatchObject({ transient: false, permanent: true });
    const r500 = await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'server error: OSError', retry: true }, 500)));
    expect(r500).toMatchObject({ transient: true, permanent: false });
    // a 5xx or a rate limit is never permanent, whatever its body says
    const r503 = await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'down', retry: false }, 503)));
    expect(r503).toMatchObject({ transient: true, permanent: false });
    const r429 = await sendEnvelope(cfg, env, fakeFetch(json({ ok: false, error: 'slow down' }, 429)));
    expect(r429).toMatchObject({ transient: true, permanent: false });
    // a proxy's HTML error page is no verdict on the record: retried with backoff as before
    const page = await sendEnvelope(cfg, env, fakeFetch(text('<html>Forbidden</html>', 403)));
    expect(page).toMatchObject({ ok: false, transient: false, permanent: false });
  });

  it('mode none or an empty endpoint never calls fetch', async () => {
    const f = fakeFetch(json({ ok: true }));
    expect((await sendEnvelope({ mode: 'none', endpoint: APPS_SCRIPT_URL, study_key: '' }, env, f)).ok).toBe(false);
    expect((await sendEnvelope(appsScriptCfg({ endpoint: '  ' }), env, f)).ok).toBe(false);
    expect(f).not.toHaveBeenCalled();
    expect(uploadEnabled({ mode: 'none', endpoint: 'https://x.org', study_key: '' })).toBe(false);
    expect(uploadEnabled(appsScriptCfg())).toBe(true);
  });
});

describe('backoff rule', () => {
  const failedAt = (attempts: number, iso: string): OutboxItem => ({
    id: 'trial:x',
    kind: 'trial',
    record_id: 'x',
    session_id: 's',
    status: 'failed',
    attempts,
    last_error: 'e',
    last_attempt_at: iso,
    sent_at: null,
    confirmed: false,
  });

  it('is min(2^attempts · 5 s, 10 min), jittered by 0.8–1.2', () => {
    expect(backoffDelayMs(1)).toBe(10_000);
    expect(backoffDelayMs(3)).toBe(40_000);
    expect(backoffDelayMs(7)).toBe(600_000);
    expect(backoffDelayMs(1000)).toBe(600_000);
    const t = Date.parse('2026-10-23T10:00:00.000Z');
    const item = failedAt(2, '2026-10-23T10:00:00.000Z'); // 20 s base
    expect(isDue(item, t + 15_999, () => 0)).toBe(false);
    expect(isDue(item, t + 16_000, () => 0)).toBe(true); // 0.8 × 20 s
    expect(isDue(item, t + 23_999, () => 0.9999999)).toBe(false);
    expect(isDue(item, t + 24_000, () => 0.9999999)).toBe(true); // ≈ 1.2 × 20 s
    expect(isDue({ ...item, attempts: 0 }, t, () => 0.5)).toBe(true);
    expect(isDue(item, t - 60_000, () => 0.5)).toBe(true); // clock moved backwards
  });

  it('never lets an automatic flush retry an item the server refused (permanent)', () => {
    const t = Date.parse('2026-10-23T10:00:00.000Z');
    const refused: OutboxItem = { ...failedAt(1, '2026-10-23T10:00:00.000Z'), permanent: true };
    expect(isDue(refused, t + 365 * 86_400_000, () => 0)).toBe(false);
    expect(isDue({ ...refused, attempts: 0, last_attempt_at: null }, t, () => 0)).toBe(false);
    expect(isDue({ ...refused, last_attempt_at: 'not a time' }, t, () => 0)).toBe(false);
    expect(isDue({ ...refused, permanent: false }, t + 20_000, () => 0)).toBe(true);
  });

  it('orders sessions before their trials, groups by session, oldest group first', () => {
    const it = (id: string, kind: 'session' | 'trial', session_id: string): OutboxItem => ({
      ...failedAt(0, ''),
      id,
      kind,
      session_id,
      status: 'pending',
    });
    const ordered = orderForUpload([
      it('trial:a1', 'trial', 'A'),
      it('trial:b1', 'trial', 'B'),
      it('session:A', 'session', 'A'),
      it('trial:a2', 'trial', 'A'),
      it('session:B', 'session', 'B'),
    ]);
    expect(ordered.map((i) => i.id)).toEqual(['session:A', 'trial:a1', 'trial:a2', 'session:B', 'trial:b1']);
  });
});

// ───────────────────────────── queue ─────────────────────────────

describe('upload queue', () => {
  it('sends oldest first, sessions before their trials, marking each sent + confirmed', async () => {
    const store = await freshStore();
    // trial enqueued before its session (out of order on purpose), then a second session
    await store.saveTrial(makeTrial({ trial_id: 'a-t0', session_id: 'A' }));
    await store.enqueue('trial', 'a-t0', 'A');
    await store.saveSession(makeSession({ session_id: 'A' }));
    await store.enqueue('session', 'A', 'A');
    await seedSession(store, 'B', 1);

    const f = fakeFetch(json({ ok: true }));
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    const report = await q.flush();

    expect(sentEnvelopes(f).map((e) => `${e.kind}:${'trial_id' in e.record ? e.record.trial_id : e.record.session_id}`)).toEqual([
      'session:A',
      'trial:a-t0',
      'session:B',
      'trial:B-t0',
    ]);
    expect(report).toEqual({ attempted: 4, sent: 4, failed: 0, refused: 0, counts: { pending: 0, sent: 4, failed: 0 } });
    const items = await store.listOutbox();
    expect(items.every((i) => i.status === 'sent' && i.confirmed && i.attempts === 1 && i.sent_at !== null)).toBe(true);
    for (const e of sentEnvelopes(f)) expectValid('UploadEnvelope', e);
  });

  it('a TypeError marks the item failed (attempts++), keeps it queued and stops the run', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 2);
    const f = fakeFetch(networkError());
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    const report = await q.flush();

    expect(f).toHaveBeenCalledTimes(1); // the other items would fail the same way
    expect(report).toMatchObject({ attempted: 1, sent: 0, failed: 1, refused: 0, counts: { pending: 2, sent: 0, failed: 1 } });
    const item = await byId(store, 'session:S');
    expect(item).toMatchObject({ status: 'failed', attempts: 1, confirmed: false });
    expect(item).not.toHaveProperty('permanent');
    expect(item?.last_error).toMatch(/network error/);
    expect(item?.last_attempt_at).not.toBeNull();
  });

  it('HTTP 500 is a failure that stops the run; a per-record refusal ({ok:false}) does not', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 2);
    const f500 = fakeFetch(text('down', 500));
    const r1 = await createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f500 }).flush();
    expect(r1).toMatchObject({ attempted: 1, failed: 1, refused: 0 });
    expect((await byId(store, 'session:S'))?.last_error).toBe('HTTP 500');
    expect(await byId(store, 'session:S')).not.toHaveProperty('permanent');

    const store2 = await freshStore();
    await seedSession(store2, 'S', 2);
    const rejecting = fakeFetch(json({ ok: false, error: 'bad record', retry: false }), json({ ok: true }));
    const r2 = await createUploadQueue(store2, () => appsScriptCfg(), { fetchImpl: rejecting }).flush();
    expect(r2).toMatchObject({ attempted: 3, sent: 2, failed: 1, refused: 1 });
    expect(await byId(store2, 'session:S')).toMatchObject({ status: 'failed', attempts: 1, permanent: true });
    expect((await byId(store2, 'session:S'))?.last_error).toContain('bad record');
  });

  it('respects the jittered backoff, and flush({force:true}) bypasses it', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const f = fakeFetch(networkError(), networkError(), json({ ok: true }));
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock, random: () => 0.5 });

    expect((await q.flush()).failed).toBe(1); // attempts = 1 → wait 10 s × (0.8 + 0.4 × 0.5) = 10 s
    clock += 9_999;
    expect((await q.flush()).attempted).toBe(0);
    clock += 1;
    expect((await q.flush()).failed).toBe(1); // attempts = 2 → wait 20 s
    clock += 5_000;
    expect((await q.flush()).attempted).toBe(0);
    const forced = await q.flush({ force: true });
    expect(forced).toMatchObject({ attempted: 1, sent: 1 });
    expect(await byId(store, 'session:S')).toMatchObject({ status: 'sent', attempts: 3, last_error: null });
    expect(f).toHaveBeenCalledTimes(3);
  });

  it('concurrent flush() calls share one in-flight run', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    const gate = deferred<void>();
    const f = vi.fn(async () => {
      await gate.promise;
      return json({ ok: true })();
    });
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    const p1 = q.flush();
    const p2 = q.flush();
    expect(p2).toBe(p1);
    gate.resolve();
    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toBe(r2);
    expect(f).toHaveBeenCalledTimes(2); // session + trial, each exactly once
    expect(r1.sent).toBe(2);
    // the next flush is a new run with nothing left to do
    expect((await q.flush()).attempted).toBe(0);
  });

  it('a forced flush during a normal run makes the running pass ignore backoff', async () => {
    const store = await freshStore();
    await seedSession(store, 'A', 0);
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const q0 = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: fakeFetch(networkError()), now: () => clock });
    await q0.flush(); // session:A is now failed and backing off
    await seedSession(store, 'B', 0);

    const gate = deferred<void>();
    const f = vi.fn(async () => {
      await gate.promise;
      return json({ ok: true })();
    });
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock });
    const running = q.flush(); // skips A (backoff), uploads B
    const forced = q.flush({ force: true });
    expect(forced).toBe(running);
    gate.resolve();
    const report = await running;
    expect(report.sent).toBe(2); // B, then A on the next pass
    expect((await byId(store, 'session:A'))?.status).toBe('sent');
  });

  it('drops items whose record no longer exists', async () => {
    const store = await freshStore();
    await store.enqueue('trial', 'never-saved', 'S');
    await seedSession(store, 'S', 0);
    const f = fakeFetch(json({ ok: true }));
    const report = await createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f }).flush();
    expect(report).toMatchObject({ attempted: 1, sent: 1, counts: { pending: 0, sent: 1, failed: 0 } });
    expect((await store.listOutbox()).map((i) => i.id)).toEqual(['session:S']);
  });

  it('drops missing records also with a store that lacks deleteOutbox (marks them, unconfirmed)', async () => {
    const real = await freshStore();
    const { deleteOutbox: _unused, ...withoutDelete } = real;
    await real.enqueue('trial', 'never-saved', 'S');
    const f = fakeFetch(json({ ok: true }));
    await createUploadQueue(withoutDelete, () => appsScriptCfg(), { fetchImpl: f }).flush();
    expect(f).not.toHaveBeenCalled();
    expect(await byId(real, 'trial:never-saved')).toMatchObject({ status: 'sent', confirmed: false });
  });

  it("mode 'none' is a no-op that still reports the remaining counts", async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 2);
    const f = fakeFetch(json({ ok: true }));
    const q = createUploadQueue(store, () => ({ mode: 'none', endpoint: '', study_key: '' }), { fetchImpl: f });
    expect(await q.flush()).toEqual({ attempted: 0, sent: 0, failed: 0, refused: 0, counts: { pending: 3, sent: 0, failed: 0 } });
    expect(f).not.toHaveBeenCalled();
  });

  it('json_post mode sends application/json; apps_script mode sends no headers', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    const f = fakeFetch(json({ ok: true }));
    await createUploadQueue(store, () => ({ mode: 'json_post', endpoint: '/api/ingest', study_key: 'k' }), {
      fetchImpl: f,
    }).flush();
    for (const [, init] of f.mock.calls as FetchArgs[]) {
      expect(init?.headers).toEqual({ 'Content-Type': 'application/json' });
    }

    const store2 = await freshStore();
    await seedSession(store2, 'S', 1);
    const g = fakeFetch(json({ ok: true }));
    await createUploadQueue(store2, () => appsScriptCfg(), { fetchImpl: g }).flush();
    expect(g).toHaveBeenCalledTimes(2);
    for (const [, init] of g.mock.calls as FetchArgs[]) expect(init).not.toHaveProperty('headers');
  });

  it('applies redactSession to session records only, without touching the stored copy', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    const f = fakeFetch(json({ ok: true }));
    const redactSession = vi.fn((s: SessionRecord): SessionRecord => ({ ...s, participant: { ...s.participant, nickname: null } }));
    await createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, redactSession }).flush();
    expect(redactSession).toHaveBeenCalledTimes(1);
    const [sessionEnv, trialEnv] = sentEnvelopes(f);
    expect(sessionEnv.kind).toBe('session');
    expect((sessionEnv.record as SessionRecord).participant.nickname).toBeNull();
    expect(sessionEnv.study_key).toBe('k-123');
    expect(trialEnv.kind).toBe('trial');
    expect((await store.getSession('S'))?.participant.nickname).toBe('Dana');
  });

  it('re-sends a record that was re-enqueued while its upload was in flight', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    const gate = deferred<void>();
    let calls = 0;
    const f = vi.fn(async () => {
      calls += 1;
      if (calls === 1) await gate.promise;
      return json({ ok: true })();
    });
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    const run = q.flush();
    await vi.waitFor(() => expect(calls).toBe(1));
    // the session finishes while v1 is being uploaded
    await store.saveSession(makeSession({ session_id: 'S', status: 'completed' }));
    await store.enqueue('session', 'S', 'S');
    gate.resolve();
    const report = await run;

    expect(report.sent).toBe(2);
    const bodies = f.mock.calls.map((c) => JSON.parse(String((c as unknown as FetchArgs)[1]?.body)) as UploadEnvelope);
    expect((bodies[1].record as SessionRecord).status).toBe('completed');
    expect(await byId(store, 'session:S')).toMatchObject({ status: 'sent', confirmed: true });
  });

  it('does nothing while the browser reports offline, unless forced', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    vi.stubGlobal('navigator', { onLine: false });
    const f = fakeFetch(json({ ok: true }));
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    expect((await q.flush()).attempted).toBe(0);
    expect((await q.flush({ force: true })).sent).toBe(1);
  });

  it('survives a throwing getConfig', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    const q = createUploadQueue(store, () => {
      throw new Error('no config');
    });
    expect(await q.flush()).toMatchObject({ attempted: 0, counts: { pending: 1, sent: 0, failed: 0 } });
  });

  it('onChange receives fresh counts after state changes and can unsubscribe', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    const f = fakeFetch(json({ ok: true }), networkError());
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    const seen: string[] = [];
    const off = q.onChange((c) => seen.push(`${c.pending}/${c.sent}/${c.failed}`));
    await q.flush();
    expect(seen).toEqual(['2/0/0', '1/1/0', '0/1/1']);
    off();
    await store.enqueue('trial', 'S-t0', 'S');
    await q.flush({ force: true });
    expect(seen).toHaveLength(3);
  });
});

describe('upload queue: records the server refused (permanent)', () => {
  it('are marked failed + permanent with the reason, counted as refused, and the run goes on', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 2);
    const f = fakeFetch(refusal());
    const report = await createUploadQueue(store, () => appsScriptCfg({ study_key: 'old-key' }), { fetchImpl: f }).flush();
    expect(report).toMatchObject({ attempted: 3, sent: 0, failed: 3, refused: 3, counts: { pending: 0, sent: 0, failed: 3 } });
    for (const item of await store.listOutbox()) {
      expect(item).toMatchObject({ status: 'failed', attempts: 1, confirmed: false, permanent: true });
      expect(item.last_error).toContain('study key does not match');
    }
  });

  it('are never retried by an automatic flush, however long ago they were refused', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const f = fakeFetch(refusal());
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock, random: () => 0 });
    await q.flush();
    expect(f).toHaveBeenCalledTimes(2);
    for (const step of [60_000, 600_000, 86_400_000, 30 * 86_400_000]) {
      clock += step;
      expect(await q.flush()).toMatchObject({ attempted: 0, failed: 0 });
    }
    expect(f).toHaveBeenCalledTimes(2);
    // ...while a transiently failed record next to them is still retried with its backoff
    await seedSession(store, 'T', 0);
    const g = fakeFetch(networkError(), json({ ok: true }));
    const q2 = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: g, now: () => clock, random: () => 0.5 });
    expect(await q2.flush()).toMatchObject({ attempted: 1, failed: 1, refused: 0 });
    clock += 9_999;
    expect((await q2.flush()).attempted).toBe(0);
    clock += 1;
    expect(await q2.flush()).toMatchObject({ attempted: 1, sent: 1 });
    expect(g).toHaveBeenCalledTimes(2);
  });

  it('"Upload now" (flush({force: true})) sends them again and clears the flag on success', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 1);
    let key = 'old-key';
    const f = fakeFetch(refusal());
    const q = createUploadQueue(store, () => appsScriptCfg({ study_key: key }), { fetchImpl: f });
    expect(await q.flush()).toMatchObject({ attempted: 2, refused: 2 });
    key = 'new-key'; // the page was reloaded and loaded the new config.json
    f.mockImplementation(async () => json({ ok: true })());
    expect((await q.flush()).attempted).toBe(0); // still not on its own
    const forced = await q.flush({ force: true });
    expect(forced).toEqual({ attempted: 2, sent: 2, failed: 0, refused: 0, counts: { pending: 0, sent: 2, failed: 0 } });
    for (const item of await store.listOutbox()) {
      expect(item).toMatchObject({ status: 'sent', confirmed: true, attempts: 2, last_error: null });
      expect(item).not.toHaveProperty('permanent');
    }
    expect(sentEnvelopes(f).map((e) => e.study_key)).toEqual(['old-key', 'old-key', 'new-key', 'new-key']);
  });

  it('a forced retry that is refused again stays permanent; one that fails transiently returns to the automatic retries', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const f = fakeFetch(refusal(), refusal(), networkError(), json({ ok: true }));
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock, random: () => 0.5 });
    await q.flush();
    expect(await q.flush({ force: true })).toMatchObject({ attempted: 1, refused: 1 });
    expect(await byId(store, 'session:S')).toMatchObject({ status: 'failed', attempts: 2, permanent: true });

    expect(await q.flush({ force: true })).toMatchObject({ attempted: 1, failed: 1, refused: 0 }); // offline this time
    const item = await byId(store, 'session:S');
    expect(item).toMatchObject({ status: 'failed', attempts: 3 });
    expect(item).not.toHaveProperty('permanent');
    expect(item?.last_error).toMatch(/network error/);
    // back in the automatic cycle, with its backoff: 2^3 · 5 s = 40 s
    clock += 39_999;
    expect((await q.flush()).attempted).toBe(0);
    clock += 1;
    expect(await q.flush()).toMatchObject({ attempted: 1, sent: 1 });
  });

  it('a re-enqueued record (new content) is a fresh pending item that the automatic flushes send', async () => {
    const store = await freshStore();
    await seedSession(store, 'S', 0);
    const f = fakeFetch(json({ ok: false, error: 'bad record', retry: false }), json({ ok: true }));
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f });
    await q.flush();
    expect(await byId(store, 'session:S')).toMatchObject({ permanent: true });
    await store.saveSession(makeSession({ session_id: 'S', status: 'completed' }));
    await store.enqueue('session', 'S', 'S');
    const item = await byId(store, 'session:S');
    expect(item).toMatchObject({ status: 'pending', attempts: 0 });
    expect(item).not.toHaveProperty('permanent');
    expect(await q.flush()).toMatchObject({ attempted: 1, sent: 1 });
  });
});

describe('upload queue triggers (start/stop)', () => {
  it('never retry a refused record: not on start, online, visibility or the 60 s tick', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] }); // not setImmediate: IndexedDB needs it
    const win = new EventTarget();
    const doc = Object.assign(new EventTarget(), { visibilityState: 'visible' as DocumentVisibilityState });
    vi.stubGlobal('window', win);
    vi.stubGlobal('document', doc);

    const store = await freshStore();
    await seedSession(store, 'S', 0);
    const f = fakeFetch(refusal());
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock, random: () => 0 });

    q.start();
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(1));
    await drain();
    expect(await byId(store, 'session:S')).toMatchObject({ status: 'failed', permanent: true });

    clock += 86_400_000; // a day later: far beyond any backoff
    win.dispatchEvent(new Event('online'));
    await drain();
    doc.dispatchEvent(new Event('visibilitychange'));
    await drain();
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    await drain();
    expect(f).toHaveBeenCalledTimes(1);
    q.stop();

    // a new start (the page was reloaded) does not retry it either: that needs "Upload now"
    const q2 = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock });
    q2.start();
    await drain();
    expect(f).toHaveBeenCalledTimes(1);
    expect(await q2.flush({ force: true })).toMatchObject({ attempted: 1 });
    q2.stop();
  });

  it("flushes on start, on 'online', on visibility→visible and every 60 s while work remains", async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] }); // not setImmediate: IndexedDB needs it
    const win = new EventTarget();
    const doc = Object.assign(new EventTarget(), { visibilityState: 'visible' as DocumentVisibilityState });
    vi.stubGlobal('window', win);
    vi.stubGlobal('document', doc);

    const store = await freshStore();
    await seedSession(store, 'S', 0);
    const f = fakeFetch(networkError());
    let clock = Date.parse('2026-10-23T10:00:00.000Z');
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: f, now: () => clock, random: () => 0 });

    q.start();
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(1));
    await drain();

    win.dispatchEvent(new Event('online'));
    await drain();
    expect(f).toHaveBeenCalledTimes(1); // the item is backing off: a trigger is not a forced flush

    clock += 60_000;
    win.dispatchEvent(new Event('online'));
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(2));
    await drain();

    clock += 120_000;
    doc.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(3));
    await drain();

    clock += 600_000;
    await vi.advanceTimersByTimeAsync(60_000); // the periodic tick (a failed item remains)
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(4));
    await drain();

    q.stop();
    clock += 3_600_000;
    win.dispatchEvent(new Event('online'));
    doc.dispatchEvent(new Event('visibilitychange'));
    await vi.advanceTimersByTimeAsync(120_000);
    await drain();
    expect(f).toHaveBeenCalledTimes(4);
  });

  it('the periodic tick does nothing once the queue is empty', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    const store = await freshStore();
    const spy = vi.spyOn(store, 'listOutbox');
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: fakeFetch(json({ ok: true })) });
    q.start();
    await vi.waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    await drain();
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    await drain();
    expect(spy).toHaveBeenCalledTimes(1);
    q.stop();
  });

  it('does not flush on visibilitychange to hidden', async () => {
    const win = new EventTarget();
    const doc = Object.assign(new EventTarget(), { visibilityState: 'hidden' as DocumentVisibilityState });
    vi.stubGlobal('window', win);
    vi.stubGlobal('document', doc);
    const store = await freshStore();
    const spy = vi.spyOn(store, 'listOutbox');
    const q = createUploadQueue(store, () => appsScriptCfg(), { fetchImpl: fakeFetch(json({ ok: true })) });
    q.start();
    await vi.waitFor(() => expect(spy).toHaveBeenCalledTimes(1));
    await q.flush(); // second call (a new run)
    const before = spy.mock.calls.length;
    doc.dispatchEvent(new Event('visibilitychange'));
    await new Promise((r) => setTimeout(r, 20));
    expect(spy.mock.calls.length).toBe(before);
    q.stop();
  });
});

// ───────────────────────────── the My data screen's view of the queue ─────────────────────────────

describe('My data screen: refused items', () => {
  const item = (id: string, session_id: string, status: OutboxItem['status'], permanent?: boolean): OutboxItem => ({
    id,
    kind: 'trial',
    record_id: id,
    session_id,
    status,
    attempts: status === 'pending' ? 0 : 1,
    last_error: status === 'failed' ? 'e' : null,
    last_attempt_at: null,
    sent_at: null,
    confirmed: status === 'sent',
    ...(permanent === undefined ? {} : { permanent }),
  });

  it('counts refused items apart from the failures that are retried automatically', () => {
    const items = [
      item('a', 'S', 'sent'),
      item('b', 'S', 'failed', true),
      item('c', 'S', 'failed', true),
      item('d', 'S', 'failed'),
      item('e', 'S', 'pending'),
      item('f', 'other', 'failed', true),
      item('g', 'S', 'sent', true), // a stale flag on a sent item is not a refusal
    ];
    expect(summarizeOutbox(items, 'S')).toEqual({ total: 6, sent: 2, pending: 1, failed: 1, refused: 2 });
    expect(items.filter(isRefused).map((i) => i.id)).toEqual(['b', 'c', 'f']);
  });

  it('labels a session with refused items as refused (both languages), before any retry wording', () => {
    for (const t of [enAdmin.data.uploadState, heAdmin.data.uploadState]) {
      const refused = uploadLabel({ total: 5, sent: 2, pending: 1, failed: 0, refused: 2 }, true, t);
      expect(refused).toBe(t.refused(2));
      expect(uploadLabel({ total: 5, sent: 2, pending: 1, failed: 2, refused: 0 }, true, t)).toBe(t.failed(3));
      expect(uploadLabel({ total: 5, sent: 5, pending: 0, failed: 0, refused: 0 }, true, t)).toBe(t.done);
      expect(uploadLabel({ total: 5, sent: 2, pending: 1, failed: 0, refused: 2 }, false, t)).toBe(t.off);
    }
    expect(enAdmin.data.uploadState.refused(2)).toBe('2 refused by the server');
  });

  it('explains what to do: reload, then "Upload now" (and the export as the fallback)', () => {
    expect(enAdmin.data.refused(3)).toMatch(/^The server refused 3 items\. Reload this page .*then tap "Upload now"\..*"Export all data"/);
    expect(enAdmin.data.refused(1)).toMatch(/^The server refused 1 item\. .* If it is refused again/);
    expect(heAdmin.data.refused(3)).toContain('השרת דחה 3 פריטים.');
    expect(heAdmin.data.refused(3)).toContain('טענו מחדש את הדף');
    expect(heAdmin.data.refused(3)).toContain('„שליחה עכשיו”');
    expect(heAdmin.data.refused(1)).toContain('השרת דחה פריט אחד.');
    expect(enAdmin.data.uploadRefused(2)).toBe('2 items were refused by the server. Reload this page, then tap "Upload now" again.');
    expect(enAdmin.data.queue(1, 2, 3, 0)).toBe('3 sent · 1 waiting · 2 to retry');
    expect(enAdmin.data.queue(1, 2, 3, 4)).toBe('3 sent · 1 waiting · 2 to retry · 4 refused');
    expect(heAdmin.data.queue(1, 2, 3, 4)).toBe('נשלחו: 3 · ממתינים: 1 · לניסיון חוזר: 2 · נדחו: 4');
  });
});
