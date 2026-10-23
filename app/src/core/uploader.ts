/**
 * Upload queue: sends outbox items to the configured backend with retry and backoff.
 *
 * Transport (sendEnvelope):
 *  - mode 'apps_script': fetch(cfg.endpoint, {method: 'POST', body: JSON.stringify(envelope)}) with
 *    NO custom headers at all (a string body is sent as text/plain;charset=UTF-8, a CORS "simple
 *    request", so no preflight — Apps Script cannot answer preflights), no credentials, redirect
 *    'follow' (Apps Script runs doPost, then 302s to script.googleusercontent.com, which serves the
 *    JSON with Access-Control-Allow-Origin: *). Success = readable JSON with ok === true →
 *    {ok: true, confirmed: true}. NEVER use mode 'no-cors' (opaque = unverifiable). A TypeError
 *    ("Failed to fetch": dead deployment, uncaught server error, missing re-authorisation, content
 *    blocker, offline) or a timeout → {ok: false, error} and the item stays queued; doPost may already
 *    have stored it, which is fine because the backend overwrites by record id (idempotent).
 *  - mode 'json_post': POST with 'Content-Type: application/json' (the server must answer CORS
 *    preflights, e.g. server/receiver.py); ok iff response.ok and the body is empty or JSON whose
 *    `ok` is not false (a non-JSON body, e.g. an HTML page, is a failure: never mark data as sent on
 *    an answer we cannot understand).
 *  - mode 'none' or empty endpoint: never sends; flush() is a no-op returning remaining counts.
 *  - every request has a 45 s timeout (AbortController, plus a timer race in case an implementation
 *    ignores the signal). Never use sendBeacon/keepalive (64 KiB cap).
 *  - `confirmed` is true whenever ok is true (kept in OutboxItem for backends that cannot confirm).
 *  - `transient` (extra field) marks failures that are not about this record: network error,
 *    timeout, HTTP 5xx/429/408, a reply with retry: true (e.g. 'busy'), an unreadable reply.
 *  - `permanent` (extra field) marks the server's own refusal of this record: a JSON answer with
 *    ok: false and retry !== true (wrong study key, invalid ids, too large; the local receiver
 *    sends it with HTTP 4xx). Sending the same record again with the same settings cannot succeed.
 * Queue (UploadQueue):
 *  - flush(): sends pending + failed items oldest first, one at a time (sessions before their trials),
 *    marking each 'sent' (with confirmed flag) or 'failed' (attempts++, last_error; plus
 *    permanent: true after a refusal). Concurrent flush() calls share one in-flight run (a forced
 *    call upgrades the running one to ignore backoff). Items whose record no longer exists are
 *    dropped. A run re-lists the outbox after each pass (bounded), so items enqueued while it was
 *    uploading are not left waiting for the next trigger — e.g. the final session record saved right
 *    after the last trial.
 *    After a transient failure the run stops: the remaining items would fail the same way (and a
 *    dead endpoint would otherwise cost 45 s per item); they are retried at the next trigger.
 *    A refusal does not stop the run (it is about this record only).
 *    While navigator.onLine === false nothing is attempted (unless forced).
 *  - Refused (permanent) items are never retried automatically: a phone that still carries an old
 *    study key would otherwise re-send its whole queue every few minutes, forever. Only
 *    flush({force: true}) ("Upload now", e.g. after a reload picked up a new config.json) tries
 *    them again; success clears the flag, a transient failure hands the item back to the automatic
 *    retries, another refusal keeps it. A re-enqueue (new record content) starts a fresh item.
 *  - start(): flush now, on window 'online', on visibilitychange→visible, and every 60 s while
 *    failed/pending items exist; backoff per item: skip if attempts ≥ 1 and the last attempt was less
 *    than min(2^attempts · 5 s, 10 min) · (0.8 + 0.4·random) ago (jittered; flush({force:true})
 *    ignores backoff). One request in flight at a time per device (Apps Script allows ~30
 *    simultaneous executions per script owner in total).
 *  - onChange(cb) is called with fresh counts after every state change.
 *  - attempts counts every attempt (also the successful one); a re-enqueue resets it to 0.
 *
 * IMPLEMENTED BY: data agent.
 */
import type { OutboxItem, SessionRecord, TrialRecord, UploadConfig, UploadEnvelope } from './types';
import type { LocalStore, OutboxCounts } from './storage';

export const UPLOAD_TIMEOUT_MS = 45_000;
export const FLUSH_INTERVAL_MS = 60_000;
export const BACKOFF_BASE_MS = 5_000;
export const BACKOFF_MAX_MS = 10 * 60_000;
/** Upper bound of outbox re-listings within one flush run (guards against an enqueue loop). */
const MAX_PASSES = 5;

export interface SendResult {
  ok: boolean;
  confirmed: boolean;
  error: string | null;
  /** true when the failure is not specific to this record (see top): the queue stops its run */
  transient?: boolean;
  /** true when the server refused this record (see top): only a forced flush tries it again */
  permanent?: boolean;
}

/** true when the config can actually upload (a known mode and a non-empty endpoint). */
export function uploadEnabled(cfg: UploadConfig | null | undefined): boolean {
  return (
    !!cfg &&
    (cfg.mode === 'apps_script' || cfg.mode === 'json_post') &&
    typeof cfg.endpoint === 'string' &&
    cfg.endpoint.trim() !== ''
  );
}

/** min(2^attempts · 5 s, 10 min): the un-jittered wait after `attempts` failed attempts. */
export function backoffDelayMs(attempts: number): number {
  return Math.min(2 ** Math.max(0, attempts) * BACKOFF_BASE_MS, BACKOFF_MAX_MS);
}

/**
 * Whether an automatic flush may attempt an item now: never one the server refused (permanent),
 * otherwise by the backoff rule at the top. A forced flush does not ask.
 */
export function isDue(item: OutboxItem, nowMs: number, random: () => number = Math.random): boolean {
  if (item.permanent) return false;
  if (item.attempts < 1 || !item.last_attempt_at) return true;
  const last = Date.parse(item.last_attempt_at);
  // unreadable time, or the device clock moved backwards: do not wait (possibly forever)
  if (!Number.isFinite(last) || last > nowMs) return true;
  return nowMs - last >= backoffDelayMs(item.attempts) * (0.8 + 0.4 * random());
}

export function buildEnvelope(
  kind: 'session' | 'trial',
  record: SessionRecord | TrialRecord,
  cfg: UploadConfig,
): UploadEnvelope {
  return {
    kind,
    study_key: cfg.study_key ? cfg.study_key : null,
    sent_at: new Date().toISOString(),
    record,
  };
}

function failure(error: string, transient: boolean, permanent = false): SendResult {
  return { ok: false, confirmed: false, error, transient, permanent };
}

const SUCCESS: SendResult = { ok: true, confirmed: true, error: null, transient: false, permanent: false };

function errorName(e: unknown): string {
  return e && typeof e === 'object' && typeof (e as { name?: unknown }).name === 'string'
    ? (e as { name: string }).name
    : '';
}

function errorMessage(e: unknown): string {
  if (e && typeof e === 'object' && typeof (e as { message?: unknown }).message === 'string') {
    return (e as { message: string }).message;
  }
  return String(e);
}

function snippet(text: string): string {
  const s = text.replace(/\s+/g, ' ').trim();
  return s.length > 120 ? `${s.slice(0, 120)}…` : s;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function interpretResponse(mode: UploadConfig['mode'], status: number, httpOk: boolean, text: string): SendResult {
  const body = parseJson(text);
  const serverError = isObject(body) && typeof body.error === 'string' ? body.error : null;
  // the server's own verdict on this record: refused, and not asked to retry
  const refused = isObject(body) && body.ok === false && body.retry !== true;
  if (!httpOk) {
    const transient = status >= 500 || status === 429 || status === 408 || status === 0;
    return failure(`HTTP ${status}${serverError ? `: ${serverError}` : ''}`, transient, refused && !transient);
  }
  if (isObject(body) && body.ok === false) {
    return failure(`rejected by server: ${serverError ?? 'no reason given'}`, body.retry === true, refused);
  }
  if (mode === 'apps_script') {
    // Only a readable {ok: true} proves that doPost stored the record.
    if (isObject(body) && body.ok === true) return SUCCESS;
    return failure(`unexpected response: ${snippet(text) || '(empty)'}`, true);
  }
  if (text.trim() === '' || body !== undefined) return SUCCESS;
  return failure(`unexpected response (not JSON): ${snippet(text)}`, true);
}

export async function sendEnvelope(
  cfg: UploadConfig,
  env: UploadEnvelope,
  fetchImpl?: typeof fetch,
): Promise<SendResult> {
  if (!uploadEnabled(cfg)) return failure('upload disabled (mode none or no endpoint)', false);
  const doFetch = fetchImpl ?? (typeof fetch === 'function' ? fetch : undefined);
  if (!doFetch) return failure('fetch is not available', true);

  let body: string;
  try {
    body = JSON.stringify(env);
  } catch (e) {
    return failure(`cannot serialise the record: ${errorMessage(e)}`, false);
  }

  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      timedOut = true;
      ctrl?.abort();
      reject(new Error('timeout'));
    }, UPLOAD_TIMEOUT_MS);
  });
  timeout.catch(() => undefined); // settled by the race below or ignored

  // Apps Script: no `headers` key at all, so the request stays a CORS simple request (text/plain).
  const init: RequestInit =
    cfg.mode === 'apps_script'
      ? { method: 'POST', body, mode: 'cors', credentials: 'omit', redirect: 'follow' }
      : {
          method: 'POST',
          body,
          headers: { 'Content-Type': 'application/json' },
          mode: 'cors',
          credentials: 'omit',
          redirect: 'follow',
        };
  if (ctrl) init.signal = ctrl.signal;

  try {
    const res = await Promise.race([doFetch(cfg.endpoint.trim(), init), timeout]);
    const text = await Promise.race([res.text(), timeout]);
    return interpretResponse(cfg.mode, res.status, res.ok, text);
  } catch (e) {
    if (timedOut) return failure(`timeout after ${UPLOAD_TIMEOUT_MS / 1000} s`, true);
    if (errorName(e) === 'AbortError') return failure('request aborted', true);
    if (errorName(e) === 'TypeError') {
      const hint =
        cfg.mode === 'apps_script'
          ? ' (offline, blocked, or the Apps Script deployment answered with an error page)'
          : '';
      return failure(`network error: ${errorMessage(e)}${hint}`, true);
    }
    return failure(`upload error: ${errorMessage(e)}`, true);
  } finally {
    clearTimeout(timer);
  }
}

export interface FlushReport {
  attempted: number;
  sent: number;
  failed: number;
  /** of `failed`: the items the server refused in this run (createUploadQueue always sets it) */
  refused?: number;
  counts: OutboxCounts;
}

export interface UploadQueue {
  flush(opts?: { force?: boolean }): Promise<FlushReport>;
  start(): void;
  stop(): void;
  onChange(cb: (c: OutboxCounts) => void): () => void;
}

export interface UploadQueueOptions {
  fetchImpl?: typeof fetch;
  /** applied to session records' participant snapshot before upload */
  redactSession?: (s: SessionRecord) => SessionRecord;
  /** clock (epoch ms) for backoff decisions; tests inject it */
  now?: () => number;
  /** [0,1) source for the backoff jitter; tests inject it */
  random?: () => number;
}

/**
 * Upload order: oldest first, grouped by session (a group's place = its oldest item) with the
 * session record ahead of its trials, so the backend sees a session before its trials.
 */
export function orderForUpload(items: OutboxItem[]): OutboxItem[] {
  const groups = new Map<string, { sessions: OutboxItem[]; trials: OutboxItem[] }>();
  for (const item of items) {
    let g = groups.get(item.session_id);
    if (!g) {
      g = { sessions: [], trials: [] };
      groups.set(item.session_id, g);
    }
    (item.kind === 'session' ? g.sessions : g.trials).push(item);
  }
  return [...groups.values()].flatMap((g) => [...g.sessions, ...g.trials]);
}

function isOffline(): boolean {
  const nav = (globalThis as { navigator?: { onLine?: unknown } }).navigator;
  return !!nav && nav.onLine === false;
}

const EMPTY_COUNTS: OutboxCounts = { pending: 0, sent: 0, failed: 0 };

export function createUploadQueue(
  store: LocalStore,
  getConfig: () => UploadConfig,
  opts: UploadQueueOptions = {},
): UploadQueue {
  const now = opts.now ?? (() => Date.now());
  const random = opts.random ?? Math.random;
  const listeners = new Set<(c: OutboxCounts) => void>();
  let lastCounts: OutboxCounts | null = null;
  let running: { promise: Promise<FlushReport>; state: { force: boolean } } | null = null;
  let timer: ReturnType<typeof setInterval> | null = null;
  let detach: (() => void) | null = null;

  const emit = (c: OutboxCounts): void => {
    const same =
      lastCounts !== null &&
      lastCounts.pending === c.pending &&
      lastCounts.sent === c.sent &&
      lastCounts.failed === c.failed;
    lastCounts = c;
    if (same) return;
    for (const cb of [...listeners]) {
      try {
        cb(c);
      } catch (e) {
        console.error('upload queue: onChange listener failed', e);
      }
    }
  };

  const refreshCounts = async (): Promise<OutboxCounts> => {
    try {
      const c = await store.outboxCounts();
      emit(c);
      return c;
    } catch {
      return lastCounts ?? { ...EMPTY_COUNTS };
    }
  };

  const config = (): UploadConfig | null => {
    try {
      return getConfig();
    } catch {
      return null;
    }
  };

  const loadRecord = async (item: OutboxItem): Promise<SessionRecord | TrialRecord | undefined> =>
    item.kind === 'session' ? store.getSession(item.record_id) : store.getTrial(item.record_id);

  const drop = async (item: OutboxItem): Promise<void> => {
    if (store.deleteOutbox) {
      await store.deleteOutbox(item.id);
      return;
    }
    // stores without deleteOutbox: take the entry out of the queue without claiming an upload
    await store.updateOutbox({
      ...item,
      status: 'sent',
      confirmed: false,
      last_error: 'record no longer exists (dropped)',
      last_attempt_at: new Date(now()).toISOString(),
    });
  };

  const run = async (state: { force: boolean }): Promise<FlushReport> => {
    let attempted = 0;
    let sent = 0;
    let failed = 0;
    let refused = 0;
    await refreshCounts();
    try {
      const done = new Set<string>();
      for (let pass = 0; pass < MAX_PASSES; pass++) {
        if (!uploadEnabled(config())) break;
        if (!state.force && isOffline()) break;
        const queue = orderForUpload((await store.listOutbox()).filter((i) => i.status !== 'sent'));
        let progressed = false;
        let halted = false;
        for (const item of queue) {
          // attempted earlier in this run: only again if it was re-enqueued since (new content)
          const fresh = item.status === 'pending' && item.attempts === 0;
          if (done.has(item.id) && !fresh) continue;
          // backoff, and refused (permanent) items: only a forced flush sends those
          if (!state.force && !isDue(item, now(), random)) continue;
          const cfg = config();
          if (!cfg || !uploadEnabled(cfg)) {
            halted = true;
            break;
          }
          const record = await loadRecord(item);
          progressed = true;
          if (!record) {
            await drop(item);
            await refreshCounts();
            continue;
          }
          done.add(item.id);
          attempted++;
          const payload =
            item.kind === 'session' && opts.redactSession ? opts.redactSession(record as SessionRecord) : record;
          const res = await sendEnvelope(cfg, buildEnvelope(item.kind, payload, cfg), opts.fetchImpl);
          const at = new Date(now()).toISOString();
          // the flag describes the latest answer only (spreading the rest keeps the store's seq/rev)
          const { permanent: _previous, ...rest } = item;
          if (res.ok) {
            sent++;
            await store.updateOutbox({
              ...rest,
              status: 'sent',
              attempts: item.attempts + 1,
              confirmed: res.confirmed,
              last_error: null,
              last_attempt_at: at,
              sent_at: at,
            });
          } else {
            failed++;
            if (res.permanent) refused++;
            await store.updateOutbox({
              ...rest,
              status: 'failed',
              attempts: item.attempts + 1,
              confirmed: false,
              last_error: res.error,
              last_attempt_at: at,
              ...(res.permanent ? { permanent: true } : {}),
            });
          }
          await refreshCounts();
          if (!res.ok && res.transient) {
            halted = true;
            break;
          }
        }
        if (halted || !progressed) break;
      }
    } catch (e) {
      // storage trouble (or a throwing redactSession): keep what was done; the next trigger retries
      console.warn('upload queue: flush interrupted', e);
    }
    const counts = await refreshCounts();
    return { attempted, sent, failed, refused, counts };
  };

  const flush = (o: { force?: boolean } = {}): Promise<FlushReport> => {
    if (running) {
      if (o.force) running.state.force = true;
      return running.promise;
    }
    const state = { force: !!o.force };
    const promise = run(state).finally(() => {
      running = null;
    });
    running = { promise, state };
    return promise;
  };

  const kick = (): void => {
    void flush();
  };

  return {
    flush,

    start() {
      if (detach) return;
      const undo: (() => void)[] = [];
      const win = (globalThis as { window?: Window }).window;
      if (win && typeof win.addEventListener === 'function') {
        win.addEventListener('online', kick);
        undo.push(() => win.removeEventListener('online', kick));
      }
      const doc = (globalThis as { document?: Document }).document;
      if (doc && typeof doc.addEventListener === 'function') {
        const onVisibility = (): void => {
          if (doc.visibilityState === 'visible') kick();
        };
        doc.addEventListener('visibilitychange', onVisibility);
        undo.push(() => doc.removeEventListener('visibilitychange', onVisibility));
      }
      timer = setInterval(() => {
        // counts are refreshed by every run, so this costs nothing when the queue is empty
        if (lastCounts === null || lastCounts.pending + lastCounts.failed > 0) kick();
      }, FLUSH_INTERVAL_MS);
      detach = () => {
        for (const u of undo) u();
        if (timer !== null) clearInterval(timer);
        timer = null;
      };
      kick();
    },

    stop() {
      detach?.();
      detach = null;
    },

    onChange(cb) {
      listeners.add(cb);
      return () => {
        listeners.delete(cb);
      };
    },
  };
}
