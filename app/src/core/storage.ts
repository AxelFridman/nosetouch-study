/**
 * Offline-first local storage (IndexedDB via `idb`). Research data lives here until uploaded and
 * stays here (for export) until the participant deletes it.
 *
 * DB "nosetouch" version 1, object stores:
 *   participants (keyPath participant_id)
 *   sessions     (keyPath session_id; index by_participant on participant_id)
 *   trials       (keyPath trial_id;   index by_session on session_id, by_participant on participant_id)
 *   outbox       (keyPath id;         index by_status on status, by_session on session_id)
 * (by_participant on trials and by_session on outbox only exist so deletes cascade completely,
 * even to orphans, without scanning the big trial records.)
 * enqueue() creates/overwrites the outbox item `${kind}:${id}` with status 'pending' (a session record
 * is re-enqueued every time it changes; the backend overwrites by id, so uploads are idempotent).
 * commit() writes a trial and/or a session AND their outbox items in one transaction (the session
 * layer's only write path for records); enqueueMissing() repairs trials without any outbox item.
 * Works in Node tests with `import 'fake-indexeddb/auto'`.
 * Robustness (iOS): WebKit can drop the IndexedDB connection ("UnknownError: Connection to Indexed
 * Database server lost") when its networking process dies, and rolls back transactions when a
 * backgrounded Safari is suspended. Keep every operation one short transaction; on a connection
 * error (UnknownError / InvalidStateError / AbortError without a cause / the db 'close' event)
 * reopen the database once and retry the operation once before rejecting. Every operation is a
 * single atomic transaction of idempotent puts/deletes, so a retry can never half-apply.
 *
 * Outbox bookkeeping: stored items carry two extra fields besides OutboxItem — `seq` (enqueue order,
 * kept when an item is re-enqueued) and `rev` (bumped by every enqueue). listOutbox() returns them;
 * callers that update an item must spread the listed item (`{...item, status: 'sent'}`) so
 * updateOutbox() can see that the record was re-enqueued while it was being uploaded and keep the
 * newer 'pending' entry instead of marking it sent. updateOutbox() never recreates a deleted item.
 * Durability: records and enqueues use {durability: 'strict'} (Chrome defaults to 'relaxed', which
 * can lose the last writes on an OS crash or power loss); status updates use the default.
 *
 * IMPLEMENTED BY: data agent.
 */
import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import { APP_VERSION, SCHEMA_VERSION } from './constants';
import type { ExportBundle, OutboxItem, ParticipantProfile, SessionRecord, TrialRecord } from './types';

export const DB_NAME = 'nosetouch';
export const DB_VERSION = 1;
/** findResumableSession() default: sessions untouched for longer are not offered for resuming. */
export const RESUME_MAX_AGE_MS = 24 * 60 * 60 * 1000;

export interface OutboxCounts {
  pending: number;
  sent: number;
  failed: number;
}

export interface LocalStore {
  saveParticipant(p: ParticipantProfile): Promise<void>;
  getParticipant(id: string): Promise<ParticipantProfile | undefined>;
  /** most recently updated first */
  listParticipants(): Promise<ParticipantProfile[]>;
  /** Deletes the participant AND all their sessions, trials and outbox items. */
  deleteParticipant(id: string): Promise<void>;

  saveSession(s: SessionRecord): Promise<void>;
  getSession(id: string): Promise<SessionRecord | undefined>;
  /** newest first (started_at desc) */
  listSessions(participantId?: string): Promise<SessionRecord[]>;
  /** Most recent 'in_progress' session of this participant updated within maxAgeMs (default 24 h). */
  findResumableSession(participantId: string, maxAgeMs?: number): Promise<SessionRecord | undefined>;
  /** Deletes the session, its trials and their outbox items. */
  deleteSession(id: string): Promise<void>;

  saveTrial(t: TrialRecord): Promise<void>;
  getTrial(id: string): Promise<TrialRecord | undefined>;
  /** ordered by trial_index, then attempt */
  listTrials(sessionId: string): Promise<TrialRecord[]>;

  /**
   * ONE atomic transaction over sessions, trials and outbox: saves the trial and/or the session and
   * enqueues each record written (as enqueue() does). Either all of it is stored or none of it, so a
   * failure (or a page killed in the middle) never leaves a trial without its session update or its
   * outbox item. The session layer saves through this (sessionFlow.ts).
   */
  commit(c: { session?: SessionRecord; trial?: TrialRecord }): Promise<void>;
  /**
   * Repairs the outbox: enqueues every stored trial that has no outbox item at all (one could be lost
   * by app versions before commit(), when the page died between the separate writes). Returns how
   * many were enqueued. Items of sent records are kept by the uploader, so this never re-sends them.
   */
  enqueueMissing(): Promise<number>;
  enqueue(kind: 'session' | 'trial', recordId: string, sessionId: string): Promise<void>;
  /** oldest enqueued first (an item keeps its place when it is re-enqueued) */
  listOutbox(status?: OutboxItem['status']): Promise<OutboxItem[]>;
  /** Writes the item if it still exists and was not re-enqueued since it was listed (see top). */
  updateOutbox(item: OutboxItem): Promise<void>;
  outboxCounts(): Promise<OutboxCounts>;
  /** Removes one outbox entry (the uploader drops entries whose record no longer exists). Optional
   * so that test doubles of LocalStore need not implement it. */
  deleteOutbox?(id: string): Promise<void>;

  /** Builds an export bundle; `redact` is applied to every participant profile and session snapshot. */
  exportBundle(opts: { sessionIds?: string[]; redact: (p: ParticipantProfile) => ParticipantProfile }): Promise<ExportBundle>;
  /** Wipes every store. */
  clearAll(): Promise<void>;
  /** navigator.storage.persist() — best effort */
  requestPersistence(): Promise<boolean>;
  estimate(): Promise<{ usage: number; quota: number } | null>;
}

/** What the outbox store actually holds. */
interface StoredOutboxItem extends OutboxItem {
  seq: number;
  rev: number;
}

interface NoseTouchDB extends DBSchema {
  participants: { key: string; value: ParticipantProfile };
  sessions: { key: string; value: SessionRecord; indexes: { by_participant: string } };
  trials: { key: string; value: TrialRecord; indexes: { by_session: string; by_participant: string } };
  outbox: { key: string; value: StoredOutboxItem; indexes: { by_status: string; by_session: string } };
}

type Db = IDBPDatabase<NoseTouchDB>;

const STRICT: IDBTransactionOptions = { durability: 'strict' };
const noop = (): void => undefined;
const OUTBOX_STATUSES: OutboxItem['status'][] = ['pending', 'sent', 'failed'];

function upgradeSchema(db: Db, oldVersion: number): void {
  if (oldVersion < 1) {
    db.createObjectStore('participants', { keyPath: 'participant_id' });
    const sessions = db.createObjectStore('sessions', { keyPath: 'session_id' });
    sessions.createIndex('by_participant', 'participant_id');
    const trials = db.createObjectStore('trials', { keyPath: 'trial_id' });
    trials.createIndex('by_session', 'session_id');
    trials.createIndex('by_participant', 'participant_id');
    const outbox = db.createObjectStore('outbox', { keyPath: 'id' });
    outbox.createIndex('by_status', 'status');
    outbox.createIndex('by_session', 'session_id');
  }
}

/** Errors after which the connection is presumed dead (see the top comment). */
export function isConnectionError(e: unknown): boolean {
  if (!e || typeof e !== 'object') return false;
  const { name, message } = e as { name?: unknown; message?: unknown };
  if (typeof message === 'string' && /connection to indexed database server lost/i.test(message)) return true;
  // AbortError only reaches us for transactions aborted without a request error (idb reports a
  // request's own error, e.g. QuotaExceededError, instead): i.e. the browser rolled it back.
  return name === 'UnknownError' || name === 'InvalidStateError' || name === 'AbortError';
}

function openConnection(name: string, onGone: () => void): Promise<Db> {
  try {
    return openDB<NoseTouchDB>(name, DB_VERSION, {
      upgrade: (db, oldVersion) => upgradeSchema(db, oldVersion),
      // A newer app version in another tab wants to upgrade: let it, and reopen on next use.
      blocking: (_cur, _next, event) => {
        (event.target as IDBDatabase | null)?.close();
        onGone();
      },
      // The browser closed the connection abnormally (the db 'close' event).
      terminated: onGone,
    });
  } catch (e) {
    // indexedDB.open itself can throw synchronously (e.g. storage disabled)
    return Promise.reject(e);
  }
}

function compareNum(a: number, b: number): number {
  return a === b ? 0 : a < b ? -1 : 1;
}

function compareStr(a: string, b: string): number {
  return a === b ? 0 : a < b ? -1 : 1;
}

/** Epoch ms of an ISO time; unparseable values sort as the oldest. */
function timeValue(iso: string | null | undefined): number {
  const v = typeof iso === 'string' ? Date.parse(iso) : NaN;
  return Number.isFinite(v) ? v : -Infinity;
}

function newestStartedFirst(a: SessionRecord, b: SessionRecord): number {
  return (
    compareNum(timeValue(b.started_at), timeValue(a.started_at)) ||
    compareNum(timeValue(b.updated_at), timeValue(a.updated_at)) ||
    compareStr(a.session_id, b.session_id)
  );
}

function oldestStartedFirst(a: SessionRecord, b: SessionRecord): number {
  return newestStartedFirst(b, a);
}

function trialOrder(a: TrialRecord, b: TrialRecord): number {
  return (
    compareNum(a.trial_index, b.trial_index) ||
    compareNum(a.attempt, b.attempt) ||
    compareNum(timeValue(a.created_at), timeValue(b.created_at)) ||
    compareStr(a.trial_id, b.trial_id)
  );
}

function outboxOrder(a: StoredOutboxItem, b: StoredOutboxItem): number {
  return compareNum(a.seq ?? 0, b.seq ?? 0) || compareStr(a.id, b.id);
}

/** A fresh 'pending' outbox item for a record; `prev` (the stored item, if any) keeps its place. */
function pendingItem(kind: 'session' | 'trial', recordId: string, sessionId: string, prev: StoredOutboxItem | undefined): StoredOutboxItem {
  return {
    id: `${kind}:${recordId}`,
    kind,
    record_id: recordId,
    session_id: sessionId,
    status: 'pending',
    attempts: 0,
    last_error: null,
    last_attempt_at: null,
    sent_at: null,
    confirmed: false,
    seq: prev?.seq ?? nextSeq(),
    rev: (prev?.rev ?? 0) + 1,
  };
}

function storageManager(): StorageManager | undefined {
  const nav = (globalThis as { navigator?: { storage?: StorageManager } }).navigator;
  return nav && typeof nav.storage === 'object' && nav.storage ? nav.storage : undefined;
}

let lastSeq = 0;
/** Enqueue order: wall-clock based so it survives reloads, strictly increasing within a page. */
function nextSeq(): number {
  lastSeq = Math.max(Date.now(), lastSeq + 1);
  return lastSeq;
}

export async function openLocalStore(dbName: string = DB_NAME): Promise<LocalStore> {
  if (typeof indexedDB === 'undefined' || !indexedDB) {
    throw new Error('IndexedDB is not available in this browser');
  }

  let current: Promise<Db> | null = null;

  const connect = (): Promise<Db> => {
    if (!current) {
      const p: Promise<Db> = openConnection(dbName, () => {
        if (current === p) current = null;
      });
      current = p;
      // a failed open must not be cached
      p.catch(() => {
        if (current === p) current = null;
      });
    }
    return current;
  };

  const discard = async (p: Promise<Db>): Promise<void> => {
    if (current === p) current = null;
    try {
      (await p).close();
    } catch {
      // never opened, or already closed
    }
  };

  /** Runs one operation; after a connection error, reopens once and retries once. */
  const run = async <T>(op: (db: Db) => Promise<T>): Promise<T> => {
    const p = connect();
    try {
      return await op(await p);
    } catch (e) {
      if (!isConnectionError(e)) throw e;
      await discard(p);
      return op(await connect());
    }
  };

  // Open eagerly so that a browser without usable IndexedDB fails at boot with a clear error.
  await run(async () => undefined);

  const listSessions = (participantId?: string): Promise<SessionRecord[]> =>
    run(async (db) => {
      const all =
        participantId === undefined
          ? await db.getAll('sessions')
          : await db.getAllFromIndex('sessions', 'by_participant', participantId);
      return all.sort(newestStartedFirst);
    });

  const store: LocalStore = {
    saveParticipant: (p) =>
      run(async (db) => {
        const tx = db.transaction('participants', 'readwrite', STRICT);
        await Promise.all([tx.store.put(p), tx.done]);
      }),

    getParticipant: (id) => run((db) => db.get('participants', id)),

    listParticipants: () =>
      run(async (db) => {
        const all = await db.getAll('participants');
        return all.sort(
          (a, b) =>
            compareNum(timeValue(b.updated_at), timeValue(a.updated_at)) ||
            compareNum(timeValue(b.created_at), timeValue(a.created_at)) ||
            compareStr(a.participant_id, b.participant_id),
        );
      }),

    deleteParticipant: (id) =>
      run(async (db) => {
        const tx = db.transaction(['participants', 'sessions', 'trials', 'outbox'], 'readwrite', STRICT);
        const sessions = tx.objectStore('sessions');
        const trials = tx.objectStore('trials');
        const outbox = tx.objectStore('outbox');
        const sessionIds = await sessions.index('by_participant').getAllKeys(id);
        // by participant AND by session: also catches trials whose session record is missing
        const trialIds = new Set(await trials.index('by_participant').getAllKeys(id));
        const outboxIds = new Set<string>();
        for (const sid of sessionIds) {
          for (const tid of await trials.index('by_session').getAllKeys(sid)) trialIds.add(tid);
          for (const oid of await outbox.index('by_session').getAllKeys(sid)) outboxIds.add(oid);
          outboxIds.add(`session:${sid}`);
        }
        for (const tid of trialIds) outboxIds.add(`trial:${tid}`);
        await Promise.all([
          tx.objectStore('participants').delete(id),
          ...sessionIds.map((k) => sessions.delete(k)),
          ...[...trialIds].map((k) => trials.delete(k)),
          ...[...outboxIds].map((k) => outbox.delete(k)),
          tx.done,
        ]);
      }),

    saveSession: (s) =>
      run(async (db) => {
        const tx = db.transaction('sessions', 'readwrite', STRICT);
        await Promise.all([tx.store.put(s), tx.done]);
      }),

    getSession: (id) => run((db) => db.get('sessions', id)),

    listSessions,

    findResumableSession: async (participantId, maxAgeMs = RESUME_MAX_AGE_MS) => {
      const now = Date.now();
      const candidates = (await listSessions(participantId)).filter((s) => {
        if (s.status !== 'in_progress') return false;
        const updated = timeValue(s.updated_at);
        // a timestamp in the future (device clock moved back) still counts as recent
        return updated !== -Infinity && now - updated <= maxAgeMs;
      });
      candidates.sort(
        (a, b) => compareNum(timeValue(b.updated_at), timeValue(a.updated_at)) || newestStartedFirst(a, b),
      );
      return candidates[0];
    },

    deleteSession: (id) =>
      run(async (db) => {
        const tx = db.transaction(['sessions', 'trials', 'outbox'], 'readwrite', STRICT);
        const trials = tx.objectStore('trials');
        const outbox = tx.objectStore('outbox');
        const trialIds = await trials.index('by_session').getAllKeys(id);
        const outboxIds = new Set(await outbox.index('by_session').getAllKeys(id));
        outboxIds.add(`session:${id}`);
        for (const tid of trialIds) outboxIds.add(`trial:${tid}`);
        await Promise.all([
          tx.objectStore('sessions').delete(id),
          ...trialIds.map((k) => trials.delete(k)),
          ...[...outboxIds].map((k) => outbox.delete(k)),
          tx.done,
        ]);
      }),

    saveTrial: (t) =>
      run(async (db) => {
        const tx = db.transaction('trials', 'readwrite', STRICT);
        await Promise.all([tx.store.put(t), tx.done]);
      }),

    getTrial: (id) => run((db) => db.get('trials', id)),

    listTrials: (sessionId) =>
      run(async (db) => (await db.getAllFromIndex('trials', 'by_session', sessionId)).sort(trialOrder)),

    commit: ({ session, trial }) =>
      run(async (db) => {
        const tx = db.transaction(['sessions', 'trials', 'outbox'], 'readwrite', STRICT);
        const outbox = tx.objectStore('outbox');
        const enqueueIn = async (kind: 'session' | 'trial', recordId: string, sessionId: string): Promise<void> => {
          const prev = await outbox.get(`${kind}:${recordId}`);
          await outbox.put(pendingItem(kind, recordId, sessionId, prev));
        };
        const ops: Promise<unknown>[] = [];
        try {
          if (trial) ops.push(tx.objectStore('trials').put(trial));
          if (session) ops.push(tx.objectStore('sessions').put(session));
          // session item first: a new session's item is older than its trials' (upload order)
          if (session) ops.push(enqueueIn('session', session.session_id, session.session_id));
          if (trial) ops.push(enqueueIn('trial', trial.trial_id, trial.session_id));
          await Promise.all([...ops, tx.done]);
        } catch (e) {
          // a put that throws synchronously (e.g. DataCloneError) does not abort the transaction by
          // itself: the requests already made would commit without the rest
          for (const op of ops) op.catch(noop);
          tx.done.catch(noop);
          try {
            tx.abort();
          } catch {
            // already finished (committed or aborted)
          }
          throw e;
        }
      }),

    enqueueMissing: () =>
      run(async (db) => {
        const tx = db.transaction(['trials', 'outbox'], 'readwrite', STRICT);
        const trials = tx.objectStore('trials');
        const outbox = tx.objectStore('outbox');
        const [trialIds, outboxIds] = await Promise.all([trials.getAllKeys(), outbox.getAllKeys()]);
        const queued = new Set(outboxIds);
        let added = 0;
        for (const id of trialIds) {
          if (queued.has(`trial:${id}`)) continue;
          const t = await trials.get(id);
          if (!t) continue;
          await outbox.put(pendingItem('trial', t.trial_id, t.session_id, undefined));
          added++;
        }
        await tx.done;
        return added;
      }),

    enqueue: (kind, recordId, sessionId) =>
      run(async (db) => {
        const tx = db.transaction('outbox', 'readwrite', STRICT);
        const prev = await tx.store.get(`${kind}:${recordId}`);
        await Promise.all([tx.store.put(pendingItem(kind, recordId, sessionId, prev)), tx.done]);
      }),

    listOutbox: (status) =>
      run(async (db) => {
        const items = status
          ? await db.getAllFromIndex('outbox', 'by_status', status)
          : await db.getAll('outbox');
        return items.sort(outboxOrder);
      }),

    updateOutbox: (item) =>
      run(async (db) => {
        const tx = db.transaction('outbox', 'readwrite');
        const prev = await tx.store.get(item.id);
        const rev = (item as Partial<StoredOutboxItem>).rev;
        // Deleted meanwhile (participant/session deleted during an upload): do not resurrect it.
        // Re-enqueued meanwhile (newer record version): the fresh 'pending' entry must win.
        if (!prev || (typeof rev === 'number' && rev !== prev.rev)) {
          await tx.done;
          return;
        }
        await Promise.all([tx.store.put({ ...item, seq: prev.seq, rev: prev.rev }), tx.done]);
      }),

    outboxCounts: () =>
      run(async (db) => {
        const tx = db.transaction('outbox', 'readonly');
        const index = tx.store.index('by_status');
        const [pending, sent, failed] = await Promise.all(OUTBOX_STATUSES.map((s) => index.count(s)));
        await tx.done;
        return { pending, sent, failed };
      }),

    deleteOutbox: (id) =>
      run(async (db) => {
        const tx = db.transaction('outbox', 'readwrite');
        await Promise.all([tx.store.delete(id), tx.done]);
      }),

    exportBundle: async ({ sessionIds, redact }) => {
      const { participants, sessions, trials } = await run(async (db) => {
        // one read-only transaction = one consistent snapshot
        const tx = db.transaction(['participants', 'sessions', 'trials'], 'readonly');
        const sessionStore = tx.objectStore('sessions');
        const trialStore = tx.objectStore('trials');
        const participantStore = tx.objectStore('participants');
        let sessions: SessionRecord[];
        let trials: TrialRecord[];
        let participants: ParticipantProfile[];
        if (sessionIds) {
          const found = await Promise.all([...new Set(sessionIds)].map((id) => sessionStore.get(id)));
          sessions = found.filter((s): s is SessionRecord => s !== undefined).sort(oldestStartedFirst);
          trials = [];
          for (const s of sessions) {
            trials.push(...(await trialStore.index('by_session').getAll(s.session_id)).sort(trialOrder));
          }
          const pids = [...new Set(sessions.map((s) => s.participant_id))];
          const profiles = await Promise.all(pids.map((id) => participantStore.get(id)));
          participants = profiles.filter((p): p is ParticipantProfile => p !== undefined);
        } else {
          // everything, including trials whose session record is missing: an export is the rescue path
          sessions = (await sessionStore.getAll()).sort(oldestStartedFirst);
          const rank = new Map(sessions.map((s, i) => [s.session_id, i]));
          trials = (await trialStore.getAll()).sort(
            (a, b) =>
              compareNum(rank.get(a.session_id) ?? Infinity, rank.get(b.session_id) ?? Infinity) ||
              compareStr(a.session_id, b.session_id) ||
              trialOrder(a, b),
          );
          participants = (await participantStore.getAll()).sort(
            (a, b) =>
              compareNum(timeValue(a.created_at), timeValue(b.created_at)) ||
              compareStr(a.participant_id, b.participant_id),
          );
        }
        await tx.done;
        return { participants, sessions, trials };
      });
      return {
        format: 'nose-touch-export',
        schema_version: SCHEMA_VERSION,
        app_version: APP_VERSION,
        exported_at: new Date().toISOString(),
        participants: participants.map((p) => redact(p)),
        sessions: sessions.map((s) => ({ ...s, participant: redact(s.participant) })),
        trials,
      };
    },

    clearAll: () =>
      run(async (db) => {
        const tx = db.transaction(['participants', 'sessions', 'trials', 'outbox'], 'readwrite', STRICT);
        await Promise.all([
          tx.objectStore('participants').clear(),
          tx.objectStore('sessions').clear(),
          tx.objectStore('trials').clear(),
          tx.objectStore('outbox').clear(),
          tx.done,
        ]);
      }),

    requestPersistence: async () => {
      const sm = storageManager();
      if (!sm) return false;
      try {
        if (typeof sm.persisted === 'function' && (await sm.persisted())) return true;
        return typeof sm.persist === 'function' ? await sm.persist() : false;
      } catch {
        return false;
      }
    },

    estimate: async () => {
      const sm = storageManager();
      if (!sm || typeof sm.estimate !== 'function') return null;
      try {
        const e = await sm.estimate();
        return { usage: e.usage ?? 0, quota: e.quota ?? 0 };
      } catch {
        return null;
      }
    },
  };

  return store;
}
