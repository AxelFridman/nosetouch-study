import 'fake-indexeddb/auto';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { isConnectionError, openLocalStore, type LocalStore } from '../src/core/storage';
import type { OutboxItem, ParticipantProfile } from '../src/core/types';
import { makeProfile, makeSession, makeTrial } from './helpers/dataFixtures';
import { expectValid } from './helpers/schema';

let dbCounter = 0;
function freshStore(): Promise<LocalStore> {
  dbCounter += 1;
  return openLocalStore(`test-storage-${dbCounter}`);
}

const isoAgo = (ms: number): string => new Date(Date.now() - ms).toISOString();
const HOUR = 3_600_000;

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('participants', () => {
  it('saves, reads, overwrites by id and lists most recently updated first', async () => {
    const store = await freshStore();
    await store.saveParticipant(makeProfile({ participant_id: 'P-AAAAAA', updated_at: '2026-10-01T00:00:00.000Z' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-BBBBBB', updated_at: '2026-10-03T00:00:00.000Z' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-CCCCCC', updated_at: '2026-10-02T00:00:00.000Z' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-AAAAAA', nickname: 'Renamed', updated_at: '2026-10-04T00:00:00.000Z' }));

    expect((await store.getParticipant('P-AAAAAA'))?.nickname).toBe('Renamed');
    expect(await store.getParticipant('P-ZZZZZZ')).toBeUndefined();
    const list = await store.listParticipants();
    expect(list.map((p) => p.participant_id)).toEqual(['P-AAAAAA', 'P-BBBBBB', 'P-CCCCCC']);
  });
});

describe('sessions and trials', () => {
  it('lists sessions newest first, optionally per participant', async () => {
    const store = await freshStore();
    await store.saveSession(makeSession({ session_id: 's-old', started_at: '2026-10-01T09:00:00.000Z' }));
    await store.saveSession(makeSession({ session_id: 's-new', started_at: '2026-10-03T09:00:00.000Z' }));
    await store.saveSession(makeSession({ session_id: 's-mid', started_at: '2026-10-02T09:00:00.000Z' }));
    await store.saveSession(
      makeSession({ session_id: 's-other', participant_id: 'P-ZZZZZZ', started_at: '2026-10-05T09:00:00.000Z' }),
    );

    expect((await store.listSessions()).map((s) => s.session_id)).toEqual(['s-other', 's-new', 's-mid', 's-old']);
    expect((await store.listSessions('P-ABCDEF')).map((s) => s.session_id)).toEqual(['s-new', 's-mid', 's-old']);
    expect((await store.listSessions('P-ZZZZZZ')).map((s) => s.session_id)).toEqual(['s-other']);
    expect(await store.getSession('nope')).toBeUndefined();
  });

  it('round-trips a trial exactly and lists trials by trial_index then attempt', async () => {
    const store = await freshStore();
    const original = makeTrial({ trial_id: 't-a', trial_index: 1, attempt: 1 });
    // inserted out of order on purpose
    await store.saveTrial(makeTrial({ trial_id: 't-c', trial_index: 2, attempt: 1 }));
    await store.saveTrial(makeTrial({ trial_id: 't-b', trial_index: 1, attempt: 2 }));
    await store.saveTrial(original);
    await store.saveTrial(makeTrial({ trial_id: 't-0', trial_index: 0, attempt: 1 }));
    await store.saveTrial(makeTrial({ trial_id: 't-x', session_id: 'other-session', trial_index: 0 }));

    const loaded = await store.getTrial('t-a');
    expect(loaded).toEqual(original);
    expectValid('TrialRecord', loaded);
    const listed = await store.listTrials('sess-0001');
    expect(listed.map((t) => [t.trial_index, t.attempt, t.trial_id])).toEqual([
      [0, 1, 't-0'],
      [1, 1, 't-a'],
      [1, 2, 't-b'],
      [2, 1, 't-c'],
    ]);
    expect(await store.listTrials('missing')).toEqual([]);
  });
});

describe('findResumableSession', () => {
  it('returns the most recently updated in-progress session within the age limit', async () => {
    const store = await freshStore();
    await store.saveSession(makeSession({ session_id: 's-done', status: 'completed', updated_at: isoAgo(1000) }));
    await store.saveSession(makeSession({ session_id: 's-stale', updated_at: isoAgo(25 * HOUR) }));
    await store.saveSession(makeSession({ session_id: 's-recent', updated_at: isoAgo(2 * HOUR) }));
    await store.saveSession(makeSession({ session_id: 's-newest', updated_at: isoAgo(HOUR) }));
    await store.saveSession(
      makeSession({ session_id: 's-other-person', participant_id: 'P-ZZZZZZ', updated_at: isoAgo(10) }),
    );

    expect((await store.findResumableSession('P-ABCDEF'))?.session_id).toBe('s-newest');
    expect((await store.findResumableSession('P-ZZZZZZ'))?.session_id).toBe('s-other-person');
    expect(await store.findResumableSession('P-YYYYYY')).toBeUndefined();
  });

  it('honours a custom maxAgeMs and ignores abandoned sessions', async () => {
    const store = await freshStore();
    await store.saveSession(makeSession({ session_id: 's-2h', updated_at: isoAgo(2 * HOUR) }));
    await store.saveSession(makeSession({ session_id: 's-gone', status: 'abandoned', updated_at: isoAgo(10) }));
    expect(await store.findResumableSession('P-ABCDEF', HOUR)).toBeUndefined();
    expect((await store.findResumableSession('P-ABCDEF', 3 * HOUR))?.session_id).toBe('s-2h');
  });

  it('ignores sessions whose updated_at cannot be parsed', async () => {
    const store = await freshStore();
    await store.saveSession(makeSession({ session_id: 's-bad', updated_at: 'not a date' }));
    expect(await store.findResumableSession('P-ABCDEF')).toBeUndefined();
  });
});

describe('outbox', () => {
  it('enqueue creates pending items; counts and status filters work', async () => {
    const store = await freshStore();
    await store.enqueue('session', 'sess-0001', 'sess-0001');
    await store.enqueue('trial', 'trial-0001', 'sess-0001');
    await store.enqueue('trial', 'trial-0002', 'sess-0001');

    const all = await store.listOutbox();
    expect(all.map((i) => i.id)).toEqual(['session:sess-0001', 'trial:trial-0001', 'trial:trial-0002']);
    expect(all[1]).toMatchObject<Partial<OutboxItem>>({
      kind: 'trial',
      record_id: 'trial-0001',
      session_id: 'sess-0001',
      status: 'pending',
      attempts: 0,
      last_error: null,
      last_attempt_at: null,
      sent_at: null,
      confirmed: false,
    });
    expect(await store.outboxCounts()).toEqual({ pending: 3, sent: 0, failed: 0 });

    await store.updateOutbox({ ...all[0], status: 'sent', confirmed: true, sent_at: '2026-10-23T10:00:00.000Z' });
    await store.updateOutbox({ ...all[1], status: 'failed', attempts: 1, last_error: 'HTTP 500' });
    expect(await store.outboxCounts()).toEqual({ pending: 1, sent: 1, failed: 1 });
    expect((await store.listOutbox('failed')).map((i) => i.id)).toEqual(['trial:trial-0001']);
    expect((await store.listOutbox('sent'))[0]).toMatchObject({ confirmed: true, sent_at: '2026-10-23T10:00:00.000Z' });
  });

  it('re-enqueue resets the item to pending but keeps its place in the queue', async () => {
    const store = await freshStore();
    await store.enqueue('session', 'sess-0001', 'sess-0001');
    await store.enqueue('trial', 'trial-0001', 'sess-0001');
    const [session] = await store.listOutbox();
    await store.updateOutbox({ ...session, status: 'failed', attempts: 3, last_error: 'timeout' });

    await store.enqueue('session', 'sess-0001', 'sess-0001');
    const after = await store.listOutbox();
    expect(after.map((i) => i.id)).toEqual(['session:sess-0001', 'trial:trial-0001']);
    expect(after[0]).toMatchObject({ status: 'pending', attempts: 0, last_error: null });
  });

  it('a stale update cannot overwrite a newer enqueue (record changed during its upload)', async () => {
    const store = await freshStore();
    await store.enqueue('session', 'sess-0001', 'sess-0001');
    const [listed] = await store.listOutbox(); // the uploader picks it up ...
    await store.enqueue('session', 'sess-0001', 'sess-0001'); // ... the session changes meanwhile
    await store.updateOutbox({ ...listed, status: 'sent', confirmed: true }); // old upload completes
    const [now] = await store.listOutbox();
    expect(now.status).toBe('pending');
    expect(await store.outboxCounts()).toEqual({ pending: 1, sent: 0, failed: 0 });
  });

  it('updateOutbox never resurrects a deleted item; deleteOutbox removes one', async () => {
    const store = await freshStore();
    await store.enqueue('trial', 'trial-0001', 'sess-0001');
    await store.enqueue('trial', 'trial-0002', 'sess-0001');
    const [first] = await store.listOutbox();
    await store.deleteOutbox!('trial:trial-0001');
    await store.updateOutbox({ ...first, status: 'sent' });
    expect((await store.listOutbox()).map((i) => i.id)).toEqual(['trial:trial-0002']);
  });
});

describe('cascading deletes', () => {
  async function seed(store: LocalStore): Promise<void> {
    await store.saveParticipant(makeProfile({ participant_id: 'P-ABCDEF' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-ZZZZZZ' }));
    for (const [sid, pid] of [
      ['s1', 'P-ABCDEF'],
      ['s2', 'P-ABCDEF'],
      ['s3', 'P-ZZZZZZ'],
    ] as const) {
      await store.saveSession(makeSession({ session_id: sid, participant_id: pid }));
      await store.enqueue('session', sid, sid);
      for (const k of [1, 2]) {
        const tid = `${sid}-t${k}`;
        await store.saveTrial(makeTrial({ trial_id: tid, session_id: sid, participant_id: pid, trial_index: k }));
        await store.enqueue('trial', tid, sid);
      }
    }
    // an orphan trial (its session record is missing) that still belongs to P-ABCDEF
    await store.saveTrial(makeTrial({ trial_id: 'orphan', session_id: 's-missing', participant_id: 'P-ABCDEF' }));
    await store.enqueue('trial', 'orphan', 's-missing');
  }

  it('deleteSession removes the session, its trials and their outbox items only', async () => {
    const store = await freshStore();
    await seed(store);
    await store.deleteSession('s1');
    expect(await store.getSession('s1')).toBeUndefined();
    expect(await store.listTrials('s1')).toEqual([]);
    expect(await store.getTrial('s1-t1')).toBeUndefined();
    const outboxIds = (await store.listOutbox()).map((i) => i.id);
    expect(outboxIds.filter((id) => id.includes('s1'))).toEqual([]);
    expect(outboxIds).toContain('session:s2');
    expect((await store.listTrials('s2')).length).toBe(2);
    expect(await store.getParticipant('P-ABCDEF')).toBeDefined();
  });

  it('deleteParticipant removes the profile, sessions, trials (orphans too) and outbox items', async () => {
    const store = await freshStore();
    await seed(store);
    await store.deleteParticipant('P-ABCDEF');
    expect(await store.getParticipant('P-ABCDEF')).toBeUndefined();
    expect(await store.listSessions('P-ABCDEF')).toEqual([]);
    for (const tid of ['s1-t1', 's1-t2', 's2-t1', 's2-t2', 'orphan']) expect(await store.getTrial(tid)).toBeUndefined();
    expect((await store.listOutbox()).map((i) => i.id)).toEqual(['session:s3', 'trial:s3-t1', 'trial:s3-t2']);
    // the other participant is untouched
    expect(await store.getParticipant('P-ZZZZZZ')).toBeDefined();
    expect((await store.listTrials('s3')).length).toBe(2);
  });

  it('clearAll wipes every store', async () => {
    const store = await freshStore();
    await seed(store);
    await store.clearAll();
    expect(await store.listParticipants()).toEqual([]);
    expect(await store.listSessions()).toEqual([]);
    expect(await store.listOutbox()).toEqual([]);
    expect(await store.outboxCounts()).toEqual({ pending: 0, sent: 0, failed: 0 });
  });
});

describe('exportBundle', () => {
  const redact = (p: ParticipantProfile): ParticipantProfile => ({ ...p, nickname: null });

  it('exports everything (chronologically) with redacted profiles and session snapshots', async () => {
    const store = await freshStore();
    await store.saveParticipant(makeProfile({ participant_id: 'P-ABCDEF', nickname: 'Dana' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-ZZZZZZ', nickname: 'Omer' }));
    await store.saveSession(makeSession({ session_id: 's-late', started_at: '2026-10-05T00:00:00.000Z' }));
    await store.saveSession(
      makeSession({
        session_id: 's-early',
        participant: makeProfile({ participant_id: 'P-ZZZZZZ', nickname: 'Omer' }),
        started_at: '2026-10-01T00:00:00.000Z',
      }),
    );
    await store.saveTrial(makeTrial({ trial_id: 't-late-1', session_id: 's-late', trial_index: 1 }));
    await store.saveTrial(makeTrial({ trial_id: 't-late-0', session_id: 's-late', trial_index: 0 }));
    await store.saveTrial(makeTrial({ trial_id: 't-early', session_id: 's-early', participant_id: 'P-ZZZZZZ' }));
    await store.saveTrial(makeTrial({ trial_id: 't-orphan', session_id: 's-gone' }));

    const bundle = await store.exportBundle({ redact });
    expectValid('ExportBundle', bundle);
    expect(bundle.format).toBe('nose-touch-export');
    expect(bundle.sessions.map((s) => s.session_id)).toEqual(['s-early', 's-late']);
    expect(bundle.trials.map((t) => t.trial_id)).toEqual(['t-early', 't-late-0', 't-late-1', 't-orphan']);
    expect(bundle.participants.map((p) => p.nickname)).toEqual([null, null]);
    expect(bundle.sessions.every((s) => s.participant.nickname === null)).toBe(true);
    // the stored copies are not modified by the redaction
    expect((await store.getParticipant('P-ABCDEF'))?.nickname).toBe('Dana');
    expect((await store.getSession('s-late'))?.participant.nickname).toBe('Dana');
  });

  it('exports only the selected sessions, their trials and their participants', async () => {
    const store = await freshStore();
    await store.saveParticipant(makeProfile({ participant_id: 'P-ABCDEF' }));
    await store.saveParticipant(makeProfile({ participant_id: 'P-ZZZZZZ' }));
    await store.saveSession(makeSession({ session_id: 's-a' }));
    await store.saveSession(makeSession({ session_id: 's-z', participant_id: 'P-ZZZZZZ' }));
    await store.saveTrial(makeTrial({ trial_id: 't-a', session_id: 's-a' }));
    await store.saveTrial(makeTrial({ trial_id: 't-z', session_id: 's-z', participant_id: 'P-ZZZZZZ' }));

    const bundle = await store.exportBundle({ sessionIds: ['s-a', 's-a', 'missing'], redact });
    expectValid('ExportBundle', bundle);
    expect(bundle.sessions.map((s) => s.session_id)).toEqual(['s-a']);
    expect(bundle.trials.map((t) => t.trial_id)).toEqual(['t-a']);
    expect(bundle.participants.map((p) => p.participant_id)).toEqual(['P-ABCDEF']);

    const empty = await store.exportBundle({ sessionIds: [], redact });
    expect(empty.sessions).toEqual([]);
    expect(empty.participants).toEqual([]);
  });
});

describe('connection loss (iOS "Connection to Indexed Database server lost")', () => {
  function failTransactions(times: number, error: () => Error): void {
    const original = IDBDatabase.prototype.transaction;
    let left = times;
    vi.spyOn(IDBDatabase.prototype, 'transaction').mockImplementation(function (this: IDBDatabase, ...args) {
      if (left > 0) {
        left -= 1;
        throw error();
      }
      return original.apply(this, args as Parameters<typeof original>);
    });
  }
  const lost = (): Error =>
    new DOMException('Connection to Indexed Database server lost. Refresh the page to try again', 'UnknownError');

  it('reopens the database once and retries the operation', async () => {
    const store = await freshStore();
    await store.saveParticipant(makeProfile());
    const openSpy = vi.spyOn(indexedDB, 'open');
    failTransactions(1, lost);
    expect((await store.getParticipant('P-ABCDEF'))?.participant_id).toBe('P-ABCDEF');
    expect(openSpy).toHaveBeenCalledTimes(1); // a fresh connection was opened
    // later operations use the new connection without reopening again
    await store.saveTrial(makeTrial());
    expect(openSpy).toHaveBeenCalledTimes(1);
  });

  it('also retries a write after InvalidStateError (connection closing)', async () => {
    const store = await freshStore();
    failTransactions(1, () => new DOMException('The database connection is closing.', 'InvalidStateError'));
    await store.saveSession(makeSession());
    expect((await store.getSession('sess-0001'))?.session_id).toBe('sess-0001');
  });

  it('rejects when the retry fails too (only one retry)', async () => {
    const store = await freshStore();
    failTransactions(2, lost);
    await expect(store.listParticipants()).rejects.toMatchObject({ name: 'UnknownError' });
    expect(await store.listParticipants()).toEqual([]); // recovered afterwards
  });

  it('does not retry errors that are not connection errors', async () => {
    const store = await freshStore();
    const openSpy = vi.spyOn(indexedDB, 'open');
    failTransactions(1, () => new DOMException('Quota exceeded', 'QuotaExceededError'));
    await expect(store.saveTrial(makeTrial())).rejects.toMatchObject({ name: 'QuotaExceededError' });
    expect(openSpy).not.toHaveBeenCalled();
  });

  it('classifies connection errors', () => {
    expect(isConnectionError(lost())).toBe(true);
    expect(isConnectionError(new DOMException('x', 'InvalidStateError'))).toBe(true);
    expect(isConnectionError(new DOMException('x', 'AbortError'))).toBe(true);
    expect(isConnectionError(new Error('Connection to Indexed Database server lost.'))).toBe(true);
    expect(isConnectionError(new DOMException('x', 'DataError'))).toBe(false);
    expect(isConnectionError(new DOMException('x', 'QuotaExceededError'))).toBe(false);
    expect(isConnectionError(null)).toBe(false);
  });
});

describe('environment', () => {
  it('fails clearly when IndexedDB does not exist', async () => {
    vi.stubGlobal('indexedDB', undefined);
    await expect(openLocalStore('no-idb')).rejects.toThrow(/IndexedDB is not available/);
  });

  it('requestPersistence and estimate use navigator.storage when present', async () => {
    const store = await freshStore();
    expect(await store.requestPersistence()).toBe(false); // Node: no navigator.storage
    expect(await store.estimate()).toBeNull();

    const persist = vi.fn(async () => true);
    vi.stubGlobal('navigator', {
      storage: { persisted: async () => false, persist, estimate: async () => ({ usage: 1234, quota: 99999 }) },
    });
    expect(await store.requestPersistence()).toBe(true);
    expect(persist).toHaveBeenCalledTimes(1);
    expect(await store.estimate()).toEqual({ usage: 1234, quota: 99999 });

    vi.stubGlobal('navigator', {
      storage: {
        persisted: async () => true,
        persist,
        estimate: async () => {
          throw new Error('blocked');
        },
      },
    });
    expect(await store.requestPersistence()).toBe(true);
    expect(persist).toHaveBeenCalledTimes(1); // already persistent: not asked again
    expect(await store.estimate()).toBeNull();
  });
});
