/**
 * backend/google-apps-script/Code.gs, run in node:vm against faithful fakes of the Apps Script
 * services (tests/helpers/infraGas.ts): envelope validation, the Drive layout, idempotent re-posts,
 * rejections (one metadata row in the quarantine sheet, capped per UTC day, never a Drive file), the
 * index sheet written under the script lock, setup()/testIngest(), and the guarantee that EVERY reply
 * is ContentService JSON (an uncaught error would reach the phone as an HTML page without CORS
 * headers, which the app can only report as "Failed to fetch").
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { sendEnvelope } from '../src/core/uploader';
import type { SessionRecord, TrialRecord } from '../src/core/types';
import { appsScriptCfg, makeEnvelope, makeProfile, makeSession, makeTrial, T0 } from './helpers/dataFixtures';
import { CODE_GS_PATH, FakeFile, FakeFolder, MANIFEST_PATH, loadAppsScript, type GasEnv } from './helpers/infraGas';
import { expectValid } from './helpers/schema';

const STUDY = 'nose-touch-v1';
const PID = 'P-ABCDEF';
const SID = 'sess-0001';
const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const SOURCE = readFileSync(CODE_GS_PATH, 'utf8');
const QUARANTINE_CAP = 500;

const sha256 = (text: string): string => createHash('sha256').update(text, 'utf8').digest('hex');

/**
 * loadAppsScript() plus Utilities.computeDigest, which the shared fakes lack. Like Apps Script it
 * returns the digest as an array of SIGNED bytes (-128..127).
 */
function load(): GasEnv {
  const env = loadAppsScript();
  Object.assign(env.globals.Utilities as object, {
    DigestAlgorithm: { MD2: 'MD2', MD5: 'MD5', SHA_1: 'SHA_1', SHA_256: 'SHA_256', SHA_384: 'SHA_384', SHA_512: 'SHA_512' },
    Charset: { US_ASCII: 'US_ASCII', UTF_8: 'UTF_8' },
    computeDigest: (algorithm: unknown, value: unknown, charset?: unknown): number[] => {
      if (algorithm !== 'SHA_256' || typeof value !== 'string' || (charset !== undefined && charset !== 'UTF_8')) {
        throw new Error('fake Utilities.computeDigest: only SHA_256 of a UTF-8 string is implemented');
      }
      return Array.from(new Int8Array(createHash('sha256').update(value, 'utf8').digest()));
    },
  });
  return env;
}

function ready(props: Record<string, string> = {}): GasEnv {
  const env = load();
  env.setup(props);
  return env;
}

/**
 * Freezes `new Date()` inside Code.gs at `iso` (the vm context has its own Date, which vitest's fake
 * timers do not reach). Returns a function that moves the frozen clock.
 */
function freezeClock(env: GasEnv, iso: string): (next: string) => void {
  env.globals.__clockMs = Date.parse(iso);
  env.evaluate(`(function () {
    const Real = Date;
    Date = class FrozenDate extends Real {
      constructor(...args) {
        if (args.length) super(...args);
        else super(globalThis.__clockMs);
      }
      static now() {
        return globalThis.__clockMs;
      }
    };
  })()`);
  return (next) => {
    env.globals.__clockMs = Date.parse(next);
  };
}

/** Drive writes of any kind: a rejection must cause none. */
const driveWrites = (env: GasEnv): string[] => env.events.filter((e) => /^drive\.(create|setContent|setTrashed)/.test(e));

function body(kind: 'session' | 'trial', record: SessionRecord | TrialRecord, key: string | null = null): string {
  return JSON.stringify(makeEnvelope(kind, record, key));
}

/** An envelope with arbitrary (possibly invalid) parts, as JSON text. */
function rawBody(over: Record<string, unknown>, record: Record<string, unknown> | null = null): string {
  return JSON.stringify({ kind: 'trial', study_key: null, sent_at: T0, record: record ?? makeTrial(), ...over });
}

function trialWith(over: Record<string, unknown>): Record<string, unknown> {
  return { ...makeTrial(), ...over };
}

function fileAt(env: GasEnv, path: string): FakeFile {
  const item = env.drive.at(path, env.rootFolder());
  if (!(item instanceof FakeFile)) throw new Error(`no live file at ${path}`);
  return item;
}

/** Every live file below the study root except the index spreadsheet. */
function dataFiles(env: GasEnv): string[] {
  const out: string[] = [];
  const walk = (folder: FakeFolder, prefix: string): void => {
    for (const child of folder.children()) {
      if (child.trashed) continue;
      if (child instanceof FakeFolder) walk(child, `${prefix}${child.name}/`);
      else if (child.mimeType !== 'application/vnd.google-apps.spreadsheet') out.push(`${prefix}${child.name}`);
    }
  };
  walk(env.rootFolder(), '');
  return out.sort();
}

function sheetOf(env: GasEnv, name: 'sessions' | 'trials' | 'quarantine') {
  const sheet = env.index().getSheetByName(name);
  if (!sheet) throw new Error(`no sheet ${name}`);
  return sheet;
}

const quarantineRows = (env: GasEnv) => sheetOf(env, 'quarantine').records();

const relevant = (events: string[]): string[] =>
  events.filter((e) => /^(lock\.|drive\.create|drive\.setContent|sheet\.appendRow|sheets\.(flush|openById))/.test(e));

// ───────────────────────────── manifest and source ─────────────────────────────

describe('appsscript.json and Code.gs source', () => {
  it('manifest: V8, Jerusalem time zone, web app executing as the owner for anonymous users, Drive + Sheets scopes only', () => {
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
    expect(manifest.timeZone).toBe('Asia/Jerusalem');
    expect(manifest.runtimeVersion).toBe('V8');
    expect(manifest.webapp).toEqual({ executeAs: 'USER_DEPLOYING', access: 'ANYONE_ANONYMOUS' });
    expect([...manifest.oauthScopes].sort()).toEqual([
      'https://www.googleapis.com/auth/drive',
      'https://www.googleapis.com/auth/spreadsheets',
    ]);
  });

  it('uses only services covered by those scopes, and no CacheService (its ~1000-item cap makes dedupe unreliable)', () => {
    for (const service of ['CacheService', 'UrlFetchApp', 'GmailApp', 'MailApp', 'HtmlService', 'ScriptApp', 'getDocumentLock']) {
      expect(SOURCE, service).not.toContain(service);
    }
  });

  it('exposes SERVER_VERSION and the editor functions', () => {
    const env = load();
    expect(env.evaluate<string>('SERVER_VERSION')).toMatch(/^\d+\.\d+\.\d+$/);
    for (const fn of ['doGet', 'doPost', 'setup', 'testIngest']) {
      expect(env.evaluate(`typeof ${fn}`), fn).toBe('function');
    }
  });
});

// ───────────────────────────── doGet ─────────────────────────────

describe('doGet (health check)', () => {
  it('answers JSON with service, version and time, and whether setup() ran', () => {
    const env = load();
    const before = env.doGet();
    expect(before.output.getMimeType()).toBe('JSON');
    expect(before.body).toEqual({
      ok: true,
      service: 'nosetouch-receiver',
      server_version: env.evaluate('SERVER_VERSION'),
      time: expect.stringMatching(ISO_RE),
      configured: false,
    });
    expect(Math.abs(Date.parse(String(before.body.time)) - Date.now())).toBeLessThan(60_000);
    env.setup();
    expect(env.doGet().body.configured).toBe(true);
  });

  it('still answers JSON when the properties service fails', () => {
    const env = load();
    env.props.failReads = new Error('Service unavailable: Properties');
    const r = env.doGet();
    expect(r.output.getMimeType()).toBe('JSON');
    expect(r.body).toMatchObject({ ok: false, service: 'nosetouch-receiver', error: expect.stringContaining('unavailable') });
  });
});

// ───────────────────────────── setup ─────────────────────────────

describe('setup()', () => {
  it('creates the root folder and the index spreadsheet inside it, stores their ids and logs their URLs', () => {
    const env = load();
    const ids = env.call<{ root_folder_id: string; index_spreadsheet_id: string }>('setup');
    const root = env.drive.at('NoseTouchStudy');
    expect(root).toBeInstanceOf(FakeFolder);
    expect(ids.root_folder_id).toBe((root as FakeFolder).id);
    expect(env.props.data.get('ROOT_FOLDER_ID')).toBe(ids.root_folder_id);
    expect(env.props.data.get('INDEX_SPREADSHEET_ID')).toBe(ids.index_spreadsheet_id);
    expect(env.props.data.get('MAX_BYTES')).toBe('2000000');
    const ssFile = env.drive.items.get(ids.index_spreadsheet_id);
    expect(ssFile?.parent).toBe(root);
    const ss = env.index();
    expect(ss.getSheets().map((s) => s.getName()).sort()).toEqual(['quarantine', 'sessions', 'trials']);
    for (const s of ss.getSheets()) {
      expect(s.getLastRow(), s.name).toBe(1); // header only
      expect(s.frozenRows).toBe(1);
    }
    expect(sheetOf(env, 'sessions').values()[0]).toEqual(expect.arrayContaining(['received_at', 'study_code', 'participant_id', 'session_id', 'status', 'file_id', 'stored_as']));
    expect(sheetOf(env, 'trials').values()[0]).toEqual(
      expect.arrayContaining(['trial_id', 'trial_index', 'attempt', 'hand', 'orientation', 'speed', 'start_position', 'variant', 'accepted', 'file_id', 'stored_as']),
    );
    // metadata and identifiers only: no file name or id (no Drive file), no error text, no body
    expect(sheetOf(env, 'quarantine').values()[0]).toEqual([
      'received_at',
      'reason',
      'bytes',
      'sha256',
      'kind',
      'record_type',
      'study_code',
      'participant_id',
      'session_id',
      'record_id',
      'schema_version',
      'app_version',
    ]);
    expect(env.logs.join('\n')).toContain((root as FakeFolder).getUrl());
    expect(env.logs.join('\n')).toContain(ss.getUrl());
    expect(env.logs.join('\n')).toMatch(/STUDY_KEY is NOT set/);
  });

  it('is safe to run again: reuses the folder and the spreadsheet', () => {
    const env = ready();
    const n = env.drive.items.size;
    const first = { ...Object.fromEntries(env.props.data) };
    env.props.data.set('STUDY_KEY', 'k-123');
    env.call('setup');
    expect(env.drive.items.size).toBe(n);
    expect(Object.fromEntries(env.props.data)).toEqual({ ...first, STUDY_KEY: 'k-123' });
    expect(env.logs.at(-1)).toMatch(/STUDY_KEY is set\./);
  });

  it('finds the existing folder and spreadsheet by name when the properties were lost', () => {
    const env = ready();
    const root = env.props.data.get('ROOT_FOLDER_ID');
    const index = env.props.data.get('INDEX_SPREADSHEET_ID');
    env.props.data.clear();
    const n = env.drive.items.size;
    env.call('setup');
    expect(env.drive.items.size).toBe(n);
    expect(env.props.data.get('ROOT_FOLDER_ID')).toBe(root);
    expect(env.props.data.get('INDEX_SPREADSHEET_ID')).toBe(index);
  });

  it('ignores a trashed folder or spreadsheet and creates fresh ones', () => {
    const env = ready();
    env.rootFolder().setTrashed(true);
    const oldIndex = env.props.data.get('INDEX_SPREADSHEET_ID')!;
    env.drive.items.get(oldIndex)!.trashed = true;
    env.call('setup');
    expect(env.rootFolder().trashed).toBe(false);
    expect(env.props.data.get('INDEX_SPREADSHEET_ID')).not.toBe(oldIndex);
    expect(env.index().getSheetByName('trials')).not.toBeNull();
  });
});

// ───────────────────────────── valid uploads ─────────────────────────────

describe('doPost: valid envelopes', () => {
  it('stores a session record (not the envelope) as <study>/<participant>/<session>/session.json', () => {
    const env = ready();
    const session = makeSession();
    const r = env.post(body('session', session));
    expect(r.output.getMimeType()).toBe('JSON');
    expect(r.body).toEqual({
      ok: true,
      kind: 'session',
      id: SID,
      stored_as: `${STUDY}/${PID}/${SID}/session.json`,
      received_at: expect.stringMatching(ISO_RE),
      server_version: env.evaluate('SERVER_VERSION'),
    });
    const file = fileAt(env, `${STUDY}/${PID}/${SID}/session.json`);
    expect(file.mimeType).toBe('text/plain');
    const stored = JSON.parse(file.content);
    expect(stored).toEqual(session);
    expectValid('SessionRecord', stored);
  });

  it('names trial files trial_<index, 3 digits>_a<attempt>_<trial_id>.json', () => {
    const env = ready();
    const cases: [number, number, string][] = [
      [0, 1, 'trial_000_a1_t-a.json'],
      [7, 2, 'trial_007_a2_t-b.json'],
      [123, 1, 'trial_123_a1_t-c.json'],
      [1234, 3, 'trial_1234_a3_t-d.json'],
    ];
    for (const [trial_index, attempt, name] of cases) {
      const trial_id = name.split('_').at(-1)!.replace('.json', '');
      const r = env.post(body('trial', makeTrial({ trial_id, trial_index, attempt })));
      expect(r.body).toMatchObject({ ok: true, kind: 'trial', id: trial_id, stored_as: `${STUDY}/${PID}/${SID}/${name}` });
      const stored = JSON.parse(fileAt(env, `${STUDY}/${PID}/${SID}/${name}`).content);
      expect(stored.trial_id).toBe(trial_id);
      expectValid('TrialRecord', stored);
    }
  });

  it('accepts application/json bodies too (curl, the local receiver protocol)', () => {
    const env = ready();
    expect(env.post(body('trial', makeTrial()), 'application/json').body.ok).toBe(true);
  });

  it('appends one index row per upload with ids and metadata only, strings stored as plain text', () => {
    const env = ready();
    const trial = makeTrial({ quality: { ...makeTrial().quality, warnings: ['low_rate', 'gap'] } });
    env.post(body('session', makeSession()));
    env.post(body('trial', trial));
    const sessions = sheetOf(env, 'sessions');
    const trials = sheetOf(env, 'trials');
    expect(sessions.getLastRow()).toBe(2);
    expect(trials.getLastRow()).toBe(2);
    const s = sessions.records()[0];
    expect(s).toMatchObject({ study_code: STUDY, participant_id: PID, session_id: SID, status: 'in_progress', preset: 'quick', n_planned_trials: 2, simulated: false, platform: 'android' });
    expect(s.stored_as).toBe(`${STUDY}/${PID}/${SID}/session.json`);
    expect(s.file_id).toBe(fileAt(env, `${STUDY}/${PID}/${SID}/session.json`).id);
    const t = trials.records()[0];
    expect(t).toMatchObject({
      trial_id: 'trial-0001',
      trial_index: 0,
      attempt: 1,
      hand: 'right',
      orientation: 'upright_toward',
      status: 'complete',
      accepted: true,
      n_motion: 3,
      warnings: 'low_rate; gap',
      stored_as: `${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`,
    });
    expect(String(t.received_at)).toMatch(ISO_RE);
    for (const sheet of [sessions, trials]) {
      const [header, row] = sheet.rows;
      expect(row.length).toBe(header.length);
      for (const cell of row) {
        expect(String(cell.value).length).toBeLessThanOrEqual(50000);
        if (typeof cell.value === 'string' && cell.value !== '') expect(cell.text, `${sheet.name}: ${cell.value}`).toBe(true);
      }
      // never a stream sample in the sheet
      const names = header.map((h) => h.value);
      for (const k of ['streams', 'motion', 't', 'acc_x', 'rr_alpha', 'participant', 'plan']) expect(names).not.toContain(k);
    }
  });

  it('stores client text as plain text, never as a formula (formula injection)', () => {
    const env = ready();
    const evil = '=HYPERLINK("https://evil.example","click")';
    env.post(body('trial', makeTrial({ app_version: evil })));
    const row = sheetOf(env, 'trials').rows[1];
    const cell = row.find((c) => c.value === evil);
    expect(cell).toMatchObject({ text: true, formula: false });
    expect(row.some((c) => c.formula)).toBe(false);
  });

  it('truncates an oversized index cell instead of failing (Sheets refuses > 50,000 characters)', () => {
    const env = ready();
    const r = env.post(body('trial', makeTrial({ app_version: 'v'.repeat(60000) })));
    expect(r.body).toMatchObject({ ok: true });
    expect(r.body.index_error).toBeUndefined();
    const longest = Math.max(...sheetOf(env, 'trials').rows[1].map((c) => String(c.value).length));
    expect(longest).toBeLessThanOrEqual(50000);
  });
});

// ───────────────────────────── idempotency ─────────────────────────────

describe('doPost: re-posts are idempotent', () => {
  it('a re-sent trial overwrites its file (one file, latest content) and logs another index row', () => {
    const env = ready();
    const first = env.post(body('trial', makeTrial({ outcome: { status: 'complete', accepted: true, participant_flag: null, note: 'first' } })));
    const second = env.post(body('trial', makeTrial({ outcome: { status: 'complete', accepted: true, participant_flag: null, note: 'second' } })));
    expect(second.body.stored_as).toBe(first.body.stored_as);
    expect(dataFiles(env)).toEqual([`${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`]);
    const file = fileAt(env, String(first.body.stored_as));
    expect(JSON.parse(file.content).outcome.note).toBe('second');
    expect(file.setContentCalls).toBe(1);
    expect(sheetOf(env, 'trials').getLastRow()).toBe(3); // header + 2 log rows (dedupe by trial_id)
  });

  it('the session record is replaced on every re-post; no duplicate folders', () => {
    const env = ready();
    env.post(body('session', makeSession({ status: 'in_progress' })));
    env.post(body('trial', makeTrial()));
    env.post(body('session', makeSession({ status: 'completed', ended_at: T0 })));
    expect(dataFiles(env)).toEqual([`${STUDY}/${PID}/${SID}/session.json`, `${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`]);
    expect(JSON.parse(fileAt(env, `${STUDY}/${PID}/${SID}/session.json`).content).status).toBe('completed');
    const names = env.drive.allFolders().map((f) => f.path());
    expect(new Set(names).size).toBe(names.length);
    expect(sheetOf(env, 'sessions').records().map((r) => r.status)).toEqual(['in_progress', 'completed']);
  });

  it('a redo (attempt 2) is a separate file next to attempt 1', () => {
    const env = ready();
    env.post(body('trial', makeTrial({ trial_id: 'tr-1', trial_index: 4, attempt: 1 })));
    env.post(body('trial', makeTrial({ trial_id: 'tr-2', trial_index: 4, attempt: 2 })));
    expect(dataFiles(env)).toEqual([`${STUDY}/${PID}/${SID}/trial_004_a1_tr-1.json`, `${STUDY}/${PID}/${SID}/trial_004_a2_tr-2.json`]);
  });

  it('collapses duplicate files left by two overlapping executions, and ignores trashed copies', () => {
    const env = ready();
    env.post(body('trial', makeTrial()));
    const folder = env.drive.at(`${STUDY}/${PID}/${SID}`, env.rootFolder()) as FakeFolder;
    const name = 'trial_000_a1_trial-0001.json';
    folder.createFile(name, '{"dup":1}', 'text/plain'); // what a racing execution leaves behind
    const trashed = folder.createFile(name, '{"old":1}', 'text/plain');
    trashed.trashed = true;
    env.post(body('trial', makeTrial()));
    const live = folder.files().filter((f) => f.name === name && !f.trashed);
    expect(live).toHaveLength(1);
    expect(JSON.parse(live[0].content).trial_id).toBe('trial-0001');
    expect(trashed.content).toBe('{"old":1}'); // trashed files are never written
  });

  it('writes into a fresh folder when the old session folder was trashed', () => {
    const env = ready();
    env.post(body('trial', makeTrial()));
    (env.drive.at(`${STUDY}/${PID}/${SID}`, env.rootFolder()) as FakeFolder).setTrashed(true);
    env.post(body('trial', makeTrial()));
    expect(dataFiles(env)).toEqual([`${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`]);
  });
});

// ───────────────────────────── rejections and quarantine ─────────────────────────────

const LONG = 'x'.repeat(65);

const INVALID: [label: string, text: string, code: string][] = [
  ['malformed JSON', '{"kind": "trial", "record": {', 'malformed_json'],
  ['a JSON array', '[]', 'bad_envelope'],
  ['a JSON string', '"hello"', 'bad_envelope'],
  ['JSON null', 'null', 'bad_envelope'],
  ['unknown kind', rawBody({ kind: 'participant' }), 'bad_kind'],
  ['no kind', rawBody({ kind: undefined }), 'bad_kind'],
  ['no record', JSON.stringify({ kind: 'trial', study_key: null, sent_at: T0 }), 'bad_record'],
  ['record is an array', rawBody({ record: [] }), 'bad_record'],
  ['kind/record_type mismatch', rawBody({ kind: 'trial' }, makeSession() as unknown as Record<string, unknown>), 'bad_record_type'],
  ['schema_version 2.0.0', rawBody({}, trialWith({ schema_version: '2.0.0' })), 'bad_schema_version'],
  ['schema_version 10.1.0', rawBody({}, trialWith({ schema_version: '10.1.0' })), 'bad_schema_version'],
  ['schema_version missing', rawBody({}, trialWith({ schema_version: undefined })), 'bad_schema_version'],
  ['study_code with a slash', rawBody({}, trialWith({ study_code: 'a/b' })), 'bad_study_code'],
  ['study_code too long', rawBody({}, trialWith({ study_code: LONG })), 'bad_study_code'],
  ['study_code empty', rawBody({}, trialWith({ study_code: '' })), 'bad_study_code'],
  ["study_code 'quarantine' (reserved)", rawBody({}, trialWith({ study_code: 'quarantine' })), 'bad_study_code'],
  ["study_code 'Quarantine' (reserved)", rawBody({}, trialWith({ study_code: 'Quarantine' })), 'bad_study_code'],
  ['participant_id with 0', rawBody({}, trialWith({ participant_id: 'P-ABC0EF' })), 'bad_participant_id'],
  ['participant_id with I', rawBody({}, trialWith({ participant_id: 'P-ABCDEI' })), 'bad_participant_id'],
  ['participant_id lower case', rawBody({}, trialWith({ participant_id: 'p-abcdef' })), 'bad_participant_id'],
  ['participant_id too long', rawBody({}, trialWith({ participant_id: 'P-ABCDEFG' })), 'bad_participant_id'],
  ['participant_id + newline', rawBody({}, trialWith({ participant_id: 'P-ABCDEF\n' })), 'bad_participant_id'],
  ['session_id with a space', rawBody({}, trialWith({ session_id: 'sess 1' })), 'bad_session_id'],
  ['session_id too long', rawBody({}, trialWith({ session_id: LONG })), 'bad_session_id'],
  ['trial_id with a dot', rawBody({}, trialWith({ trial_id: '../x' })), 'bad_trial_id'],
  ['trial_id missing', rawBody({}, trialWith({ trial_id: undefined })), 'bad_trial_id'],
  ['trial_index -1', rawBody({}, trialWith({ trial_index: -1 })), 'bad_trial_index'],
  ['trial_index 1.5', rawBody({}, trialWith({ trial_index: 1.5 })), 'bad_trial_index'],
  ["trial_index '3'", rawBody({}, trialWith({ trial_index: '3' })), 'bad_trial_index'],
  ['trial_index 2^53', rawBody({}, trialWith({ trial_index: 2 ** 53 })), 'bad_trial_index'],
  ['attempt 0', rawBody({}, trialWith({ attempt: 0 })), 'bad_attempt'],
  ["attempt '1'", rawBody({}, trialWith({ attempt: '1' })), 'bad_attempt'],
];

describe('doPost: rejections write one metadata row and no Drive file', () => {
  it.each(INVALID)('%s → {ok: false, retry: false}, one quarantine row, nothing in Drive', (_label, text, code) => {
    const env = ready();
    const items = env.drive.items.size;
    env.events.length = 0;
    const r = env.post(text);
    expect(r.output.getMimeType()).toBe('JSON');
    expect(r.body).toEqual({
      ok: false,
      error: expect.any(String),
      code,
      retry: false,
      received_at: expect.stringMatching(ISO_RE),
      server_version: env.evaluate('SERVER_VERSION'),
    });
    expect(String(r.body.error).length).toBeGreaterThan(5);
    expect(driveWrites(env)).toEqual([]);
    expect(env.drive.items.size).toBe(items);
    const rows = quarantineRows(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      received_at: r.body.received_at,
      reason: code,
      bytes: Buffer.byteLength(text, 'utf8'),
      sha256: sha256(text),
    });
    expect(env.locks.held).toBeNull();
  });

  it('the row holds the envelope identifiers only: never the record body or a profile value', () => {
    const env = ready();
    const profile = makeProfile({ nickname: 'Dana Levi', age_years: 47, height_cm: 181, phone_model: 'Galaxy S24' });
    const session = makeSession({ schema_version: '2.0.0', notes: 'private remark', participant: profile });
    const text = body('session', session);
    env.post(text);
    const row = quarantineRows(env)[0];
    expect(row).toEqual({
      received_at: expect.stringMatching(ISO_RE),
      reason: 'bad_schema_version',
      bytes: Buffer.byteLength(text, 'utf8'),
      sha256: sha256(text),
      kind: 'session',
      record_type: 'session',
      study_code: STUDY,
      participant_id: PID,
      session_id: SID,
      record_id: SID,
      schema_version: '2.0.0',
      app_version: session.app_version,
    });
    const cells = sheetOf(env, 'quarantine').values().flat();
    expect(cells).not.toContain(47);
    expect(cells).not.toContain(181);
    const joined = cells.map(String).join('\n');
    for (const secret of ['Dana Levi', 'Galaxy S24', 'private remark', 'Mozilla/5.0', '{"']) expect(joined, secret).not.toContain(secret);
    // a trial: its trial_id is the record id
    env.post(rawBody({}, trialWith({ attempt: 0 })));
    expect(quarantineRows(env)[1]).toMatchObject({ kind: 'trial', record_type: 'trial', session_id: SID, record_id: 'trial-0001' });
  });

  it('identifiers are best effort: non-text values are left out, long ones cut to 100 characters, text never becomes a formula', () => {
    const env = ready();
    env.post(rawBody({ kind: ['trial'] }, trialWith({ study_code: '=HYPERLINK("https://evil.example","x")', participant_id: { name: 'Dana' }, session_id: 's'.repeat(5000), app_version: 7 })));
    const row = quarantineRows(env)[0];
    expect(row).toMatchObject({ reason: 'bad_kind', kind: '', study_code: '=HYPERLINK("https://evil.example","x")', participant_id: '', app_version: '7' });
    expect(row.session_id).toBe(`${'s'.repeat(100)}…`);
    expect(row.record_id).toBe('trial-0001');
    const cells = sheetOf(env, 'quarantine').rows[1];
    expect(cells.some((c) => c.formula)).toBe(false);
    expect(cells.find((c) => String(c.value).startsWith('=HYPERLINK'))).toMatchObject({ text: true, formula: false });
  });

  it('a body that is not an envelope leaves only the time, reason, size and hash', () => {
    const env = ready();
    const text = '{"kind": "trial", "record": {"participant": {"nickname": "Dana"';
    env.post(text);
    expect(quarantineRows(env)[0]).toEqual({
      received_at: expect.stringMatching(ISO_RE),
      reason: 'malformed_json',
      bytes: text.length,
      sha256: sha256(text),
      kind: '',
      record_type: '',
      study_code: '',
      participant_id: '',
      session_id: '',
      record_id: '',
      schema_version: '',
      app_version: '',
    });
  });

  it('hashes the UTF-8 bytes (Apps Script returns signed bytes)', () => {
    const env = ready();
    const text = rawBody({ kind: 'nope' }, trialWith({ app_version: 'שלום ✓' }));
    env.post(text);
    expect(quarantineRows(env)[0].sha256).toBe(sha256(text));
    expect(quarantineRows(env)[0].sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('STUDY_KEY set: a missing or wrong key is rejected and logged, the right one (property trimmed) accepted', () => {
    const env = ready({ STUDY_KEY: '  k-123 ' });
    expect(env.post(body('trial', makeTrial(), null)).body).toMatchObject({ ok: false, retry: false, code: 'bad_study_key' });
    expect(env.post(body('trial', makeTrial(), 'k-999')).body).toMatchObject({ ok: false, retry: false, code: 'bad_study_key' });
    expect(dataFiles(env)).toEqual([]);
    expect(env.post(body('trial', makeTrial(), 'k-123')).body).toMatchObject({ ok: true });
    expect(quarantineRows(env).map((r) => [r.reason, r.record_id])).toEqual([
      ['bad_study_key', 'trial-0001'],
      ['bad_study_key', 'trial-0001'],
    ]);
    // the key itself is never logged
    expect(sheetOf(env, 'quarantine').values().flat()).not.toContain('k-999');
  });

  it('STUDY_KEY not set: any key (or none) is accepted', () => {
    const env = ready();
    expect(env.post(body('trial', makeTrial({ trial_id: 'a1' }), null)).body.ok).toBe(true);
    expect(env.post(body('trial', makeTrial({ trial_id: 'a2' }), 'whatever')).body.ok).toBe(true);
  });

  it('an empty or missing body is rejected and logged with 0 bytes', () => {
    const env = ready();
    env.events.length = 0;
    for (const e of [undefined, null, {}, { postData: {} }, { postData: { contents: '' } }]) {
      const r = env.postEvent(e);
      expect(r.output.getMimeType()).toBe('JSON');
      expect(r.body).toMatchObject({ ok: false, retry: false, code: 'empty_body' });
    }
    expect(quarantineRows(env).map((r) => [r.reason, r.bytes, r.sha256])).toEqual(Array(5).fill(['empty_body', 0, sha256('')]));
    expect(driveWrites(env)).toEqual([]);
  });

  it('never creates a quarantine folder, and a rejection never touches existing data files', () => {
    const env = ready();
    env.post(body('trial', makeTrial()));
    const stored = fileAt(env, `${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`).content;
    env.events.length = 0;
    for (const [, text] of INVALID) env.post(text);
    expect(driveWrites(env)).toEqual([]);
    expect(env.drive.at('quarantine', env.rootFolder())).toBeNull();
    expect(fileAt(env, `${STUDY}/${PID}/${SID}/trial_000_a1_trial-0001.json`).content).toBe(stored);
    expect(quarantineRows(env)).toHaveLength(INVALID.length);
  });
});

describe('doPost: at most 500 quarantine rows per UTC day', () => {
  it(`writes the first ${QUARANTINE_CAP} rejections of a day, then nothing; the answers stay the same`, () => {
    const env = ready();
    freezeClock(env, '2026-11-20T09:00:00.000Z');
    env.events.length = 0;
    for (let i = 0; i < QUARANTINE_CAP + 3; i++) {
      const r = env.post(rawBody({ kind: 'spam' }, trialWith({ trial_id: `t-${i}` })));
      expect(r.body).toMatchObject({ ok: false, retry: false, code: 'bad_kind' });
    }
    const rows = quarantineRows(env);
    expect(rows).toHaveLength(QUARANTINE_CAP);
    expect(rows.at(-1)?.record_id).toBe(`t-${QUARANTINE_CAP - 1}`);
    expect(env.props.data.get('QUARANTINE_COUNT')).toBe(`2026-11-20:${QUARANTINE_CAP}`);
    expect(driveWrites(env)).toEqual([]);
  });

  it('beyond the cap a rejection takes no lock and opens no spreadsheet', () => {
    const env = ready();
    freezeClock(env, '2026-11-20T09:00:00.000Z');
    env.props.data.set('QUARANTINE_COUNT', `2026-11-20:${QUARANTINE_CAP}`);
    env.events.length = 0;
    expect(env.post('{').body).toMatchObject({ ok: false, retry: false, code: 'malformed_json' });
    expect(env.events).toEqual([]);
    expect(quarantineRows(env)).toHaveLength(0);
    expect(env.props.data.get('QUARANTINE_COUNT')).toBe(`2026-11-20:${QUARANTINE_CAP}`);
  });

  it('the count is per UTC date: it starts again at 00:00 UTC (not at local midnight)', () => {
    const env = ready();
    const setClock = freezeClock(env, '2026-11-20T22:30:00.000Z'); // already 00:30 on the 21st in Israel
    env.props.data.set('QUARANTINE_COUNT', `2026-11-20:${QUARANTINE_CAP - 1}`);
    env.post('{');
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(1);
    expect(env.props.data.get('QUARANTINE_COUNT')).toBe(`2026-11-20:${QUARANTINE_CAP}`);
    setClock('2026-11-20T23:59:59.999Z');
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(1);
    setClock('2026-11-21T00:00:00.000Z');
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(2);
    expect(quarantineRows(env)[1].received_at).toBe('2026-11-21T00:00:00.000Z');
    expect(env.props.data.get('QUARANTINE_COUNT')).toBe('2026-11-21:1');
  });

  it('an unreadable counter counts as 0, and deleting it resets the day', () => {
    const env = ready();
    freezeClock(env, '2026-11-20T09:00:00.000Z');
    for (const bad of ['garbage', '2026-11-20', '2026-11-20:abc', '2026-11-20:-4']) {
      env.props.data.set('QUARANTINE_COUNT', bad);
      env.post('{');
      expect(env.props.data.get('QUARANTINE_COUNT'), bad).toBe('2026-11-20:1');
    }
    env.props.data.set('QUARANTINE_COUNT', `2026-11-20:${QUARANTINE_CAP}`);
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(4);
    env.props.data.delete('QUARANTINE_COUNT');
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(5);
  });

  it('re-checks the count under the lock: a row written by a concurrent execution counts', () => {
    const env = ready();
    freezeClock(env, '2026-11-20T09:00:00.000Z');
    env.props.data.set('QUARANTINE_COUNT', `2026-11-20:${QUARANTINE_CAP - 1}`);
    // while this execution waits for the lock, another one writes the day's last row
    env.locks.contended = (ms) => {
      if (ms === 5000) env.props.data.set('QUARANTINE_COUNT', `2026-11-20:${QUARANTINE_CAP}`);
      return false;
    };
    expect(env.post('{').body).toMatchObject({ ok: false, retry: false });
    expect(quarantineRows(env)).toHaveLength(0);
    expect(env.props.data.get('QUARANTINE_COUNT')).toBe(`2026-11-20:${QUARANTINE_CAP}`);
    expect(env.locks.held).toBeNull();
  });
});

describe('doPost: size limit (checked before parsing)', () => {
  const padded = (bytes: number): string => {
    const base = body('session', makeSession({ notes: '' }));
    return body('session', makeSession({ notes: 'n'.repeat(bytes - base.length) }));
  };

  it('rejects a body over MAX_BYTES even when it is a valid envelope, and logs only its size and hash', () => {
    const env = ready({ MAX_BYTES: '5000' });
    const text = padded(5001);
    expect(text.length).toBe(5001);
    env.events.length = 0;
    const r = env.post(text);
    expect(r.body).toMatchObject({ ok: false, retry: false, code: 'too_large' });
    expect(String(r.body.error)).toContain('5001');
    expect(driveWrites(env)).toEqual([]);
    // never parsed, so no identifiers either
    expect(quarantineRows(env)[0]).toMatchObject({ reason: 'too_large', bytes: 5001, sha256: sha256(text), kind: '', participant_id: '' });
    expect(env.post(padded(5000)).body.ok).toBe(true); // exactly at the limit is fine
  });

  it('an oversized body is never parsed: oversized garbage is too_large, not malformed_json', () => {
    const env = ready({ MAX_BYTES: '5000' });
    expect(env.post('{'.repeat(6000)).body.code).toBe('too_large');
  });

  it('defaults to 2,000,000 bytes; an oversized body is hashed, never stored', () => {
    const env = ready();
    env.props.data.delete('MAX_BYTES');
    const text = padded(2_100_000);
    const items = env.drive.items.size;
    const r = env.post(text);
    expect(r.body).toMatchObject({ ok: false, code: 'too_large' });
    expect(env.drive.items.size).toBe(items);
    expect(quarantineRows(env)[0]).toMatchObject({ bytes: 2_100_000, sha256: sha256(text) });
    const longest = Math.max(...sheetOf(env, 'quarantine').rows[1].map((c) => String(c.value).length));
    expect(longest).toBeLessThanOrEqual(64);
    expect(env.post(padded(1_999_000)).body.ok).toBe(true);
  });

  it('an unreadable MAX_BYTES falls back to the default', () => {
    for (const bad of ['abc', '0', '-5', '']) {
      const env = ready({ MAX_BYTES: bad });
      expect(env.post(padded(10_000)).body.ok, bad).toBe(true);
      expect(env.post(padded(2_000_001)).body.code, bad).toBe('too_large');
    }
  });

  it('counts bytes (postData.length), not characters', () => {
    const base = body('session', makeSession({ notes: '' })).length;
    const limit = base + 1000;
    const env = ready({ MAX_BYTES: String(limit) });
    const text = body('session', makeSession({ notes: 'ש'.repeat(750) })); // 2 bytes per character in UTF-8
    expect(text.length).toBeLessThan(limit);
    expect(Buffer.byteLength(text, 'utf8')).toBeGreaterThan(limit);
    expect(env.post(text).body.code).toBe('too_large');
  });
});

describe('doPost: before setup()', () => {
  it('a valid record is answered retry: true (the phone keeps it queued); nothing is written', () => {
    const env = load();
    const r = env.post(body('trial', makeTrial()));
    expect(r.body).toMatchObject({ ok: false, retry: true, error: expect.stringContaining('setup()') });
    expect(env.drive.items.size).toBe(1); // My Drive only
  });

  it('an invalid one is still a final rejection (nowhere to log it, nothing written)', () => {
    const env = load();
    expect(env.post('{').body).toMatchObject({ ok: false, retry: false, code: 'malformed_json' });
    expect(env.drive.items.size).toBe(1);
    expect(env.sheets.byId.size).toBe(0);
    expect(env.props.data.size).toBe(0);
  });
});

// ───────────────────────────── locks and the index ─────────────────────────────

describe('doPost: locks, index sheet and busy answers', () => {
  it('a new session: folders under the 10 s lock, the file outside any lock, the index row under the 30 s lock + flush', () => {
    const env = ready();
    env.events.length = 0;
    env.post(body('session', makeSession()));
    const base = `My Drive/NoseTouchStudy/${STUDY}`;
    expect(relevant(env.events)).toEqual([
      'lock.wait 10000',
      `drive.createFolder ${base}`,
      `drive.createFolder ${base}/${PID}`,
      `drive.createFolder ${base}/${PID}/${SID}`,
      'lock.release',
      `drive.createFile ${base}/${PID}/${SID}/session.json`,
      'sheets.openById',
      'lock.wait 30000',
      'sheet.appendRow sessions',
      'sheets.flush',
      'lock.release',
    ]);
  });

  it('later records of the session take no folder lock (fast path); a re-post overwrites', () => {
    const env = ready();
    env.post(body('session', makeSession()));
    env.events.length = 0;
    env.post(body('trial', makeTrial()));
    env.post(body('trial', makeTrial()));
    const base = `My Drive/NoseTouchStudy/${STUDY}/${PID}/${SID}`;
    expect(relevant(env.events)).toEqual([
      `drive.createFile ${base}/trial_000_a1_trial-0001.json`,
      'sheets.openById',
      'lock.wait 30000',
      'sheet.appendRow trials',
      'sheets.flush',
      'lock.release',
      `drive.setContent ${base}/trial_000_a1_trial-0001.json`,
      'sheets.openById',
      'lock.wait 30000',
      'sheet.appendRow trials',
      'sheets.flush',
      'lock.release',
    ]);
  });

  it("index lock timeout → {ok: false, error: 'busy', retry: true}; the stored file makes the retry idempotent", () => {
    const env = ready();
    env.locks.contended = (ms) => ms === 30000;
    const r = env.post(body('trial', makeTrial()));
    expect(r.body).toEqual({ ok: false, error: 'busy', retry: true, received_at: expect.stringMatching(ISO_RE), server_version: env.evaluate('SERVER_VERSION') });
    expect(dataFiles(env)).toHaveLength(1);
    expect(sheetOf(env, 'trials').getLastRow()).toBe(1);
    expect(env.locks.held).toBeNull();
    env.locks.contended = false;
    expect(env.post(body('trial', makeTrial())).body.ok).toBe(true);
    expect(dataFiles(env)).toHaveLength(1);
    expect(sheetOf(env, 'trials').getLastRow()).toBe(2);
  });

  it('folder lock timeout for a new session → busy, nothing written', () => {
    const env = ready();
    env.locks.contended = (ms) => ms === 10000;
    expect(env.post(body('session', makeSession())).body).toMatchObject({ ok: false, error: 'busy', retry: true });
    expect(dataFiles(env)).toEqual([]);
    expect(env.rootFolder().subfolders()).toEqual([]);
  });

  it('the lock is released even when appendRow throws; the stored record is still confirmed', () => {
    const env = ready();
    vi.spyOn(sheetOf(env, 'trials'), 'appendRow').mockImplementation(() => {
      throw new Error('Service Spreadsheets failed while accessing document');
    });
    const r = env.post(body('trial', makeTrial()));
    expect(r.body).toMatchObject({ ok: true, index_error: expect.stringContaining('Service Spreadsheets failed') });
    expect(env.locks.held).toBeNull();
    expect(env.sheets.flushes).toBe(0);
    expect(env.logs.join('\n')).toMatch(/index write failed/);
    expect(dataFiles(env)).toHaveLength(1);
  });

  it('a deleted or unconfigured index spreadsheet does not block uploads', () => {
    const env = ready();
    env.props.data.set('INDEX_SPREADSHEET_ID', 'deleted-id');
    expect(env.post(body('trial', makeTrial({ trial_id: 'x1' }))).body).toMatchObject({ ok: true, index_error: expect.any(String) });
    env.props.data.delete('INDEX_SPREADSHEET_ID');
    expect(env.post(body('trial', makeTrial({ trial_id: 'x2' }))).body).toMatchObject({ ok: true, index_error: expect.stringContaining('setup()') });
    expect(dataFiles(env)).toHaveLength(2);
  });

  it('a busy quarantine lock never turns a rejection into a retry; only its row is lost (and not counted)', () => {
    const env = ready();
    env.locks.contended = (ms) => ms === 5000;
    const first = env.post('{');
    expect(first.body).toEqual({
      ok: false,
      error: 'malformed JSON',
      code: 'malformed_json',
      retry: false,
      received_at: expect.stringMatching(ISO_RE),
      server_version: env.evaluate('SERVER_VERSION'),
    });
    expect(quarantineRows(env)).toHaveLength(0);
    expect(env.props.data.has('QUARANTINE_COUNT')).toBe(false);
    expect(env.logs.join('\n')).toMatch(/quarantine row not written \(malformed_json\): busy/);
    expect(env.locks.held).toBeNull();
    env.locks.contended = false;
    env.post('{');
    expect(quarantineRows(env)).toHaveLength(1);
    expect(env.props.data.get('QUARANTINE_COUNT')).toMatch(/^\d{4}-\d{2}-\d{2}:1$/);
  });

  it('a rejection whose row cannot be written (Sheets down, Logger down) is still a plain rejection', () => {
    const env = ready();
    vi.spyOn(sheetOf(env, 'quarantine'), 'appendRow').mockImplementation(() => {
      throw new Error('Service Spreadsheets failed while accessing document');
    });
    (env.globals.Logger as { log: () => never }).log = () => {
      throw new Error('logger down');
    };
    expect(env.post('{').body).toMatchObject({ ok: false, retry: false, code: 'malformed_json' });
    expect(env.locks.held).toBeNull();
    expect(env.props.data.has('QUARANTINE_COUNT')).toBe(false);
  });
});

// ───────────────────────────── always JSON ─────────────────────────────

describe('doPost: every reply is ContentService JSON', () => {
  it('root folder gone → {ok: false, retry: true, error: server error ...}', () => {
    const env = ready();
    env.props.data.set('ROOT_FOLDER_ID', 'gone');
    const r = env.post(body('trial', makeTrial()));
    expect(r.output.getMimeType()).toBe('JSON');
    expect(r.body).toMatchObject({ ok: false, retry: true, error: expect.stringMatching(/^server error: No item with the given ID/) });
  });

  it('Drive quota error while writing → retry: true', () => {
    const env = ready();
    env.drive.failWrites = new Error('Drive storage quota exceeded');
    const r = env.post(body('session', makeSession()));
    expect(r.body).toMatchObject({ ok: false, retry: true, error: expect.stringContaining('quota') });
    expect(env.locks.held).toBeNull();
  });

  it('properties service failure → retry: true', () => {
    const env = ready();
    env.props.failReads = new Error('Service invoked too many times');
    expect(env.post(body('trial', makeTrial())).body).toMatchObject({ ok: false, retry: true, error: expect.stringContaining('too many times') });
  });

  it('even when Logger itself throws inside the error path', () => {
    const env = ready();
    env.drive.failWrites = new Error('boom');
    (env.globals.Logger as { log: () => never }).log = () => {
      throw new Error('logger down');
    };
    const r = env.post(body('session', makeSession()));
    expect(r.output.getMimeType()).toBe('JSON');
    expect(r.body).toMatchObject({ ok: false, retry: true });
  });

  it('fuzz: random bodies always get a JSON reply with a boolean ok and never leave the lock held', () => {
    const env = ready({ MAX_BYTES: '20000' });
    let seed = 12345;
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    const pick = <T>(xs: T[]): T => xs[Math.floor(rnd() * xs.length)];
    const valid = JSON.parse(body('trial', makeTrial())) as Record<string, unknown>;
    const mutate = (v: unknown, depth: number): unknown => {
      if (depth > 3 || rnd() < 0.3) return pick<unknown>([null, 0, -1, 1.5, '', 'x', 'P-ABCDEF', [], {}, true, '1.0.0', 'trial', 'session']);
      if (Array.isArray(v)) return v.map((x) => mutate(x, depth + 1));
      if (v && typeof v === 'object') {
        const o: Record<string, unknown> = {};
        for (const [k, x] of Object.entries(v)) if (rnd() > 0.1) o[k] = rnd() < 0.2 ? mutate(x, depth + 1) : x;
        return o;
      }
      return v;
    };
    for (let i = 0; i < 150; i++) {
      const text = i % 5 === 0 ? JSON.stringify(valid).slice(0, Math.floor(rnd() * 400)) : JSON.stringify(mutate(valid, 0));
      const r = env.post(text ?? 'null');
      expect(r.output.getMimeType()).toBe('JSON');
      expect(typeof r.body.ok).toBe('boolean');
      if (r.body.ok === false) expect(typeof r.body.retry).toBe('boolean');
      expect(env.locks.held).toBeNull();
    }
    // whatever was refused left no file behind: Drive holds valid records in the data layout only
    const layout = /^[A-Za-z0-9_-]{1,64}\/P-[A-HJ-NP-Z2-9]{6}\/[A-Za-z0-9_-]{1,64}\/(session|trial_\d{3,}_a\d+_[A-Za-z0-9_-]{1,64})\.json$/;
    for (const path of dataFiles(env)) expect(path).toMatch(layout);
    expect(env.drive.at('quarantine', env.rootFolder())).toBeNull();
  });
});

// ───────────────────────────── testIngest ─────────────────────────────

describe('testIngest() self-test', () => {
  it('posts a session and a trial through doPost and passes after setup()', () => {
    const env = ready();
    const replies = env.call<{ ok: boolean; stored_as: string }[]>('testIngest');
    expect(replies.map((r) => r.ok)).toEqual([true, true]);
    for (const r of replies) expect(r.stored_as).toMatch(/^selftest\/P-TEST22\/selftest-\d{14}\/(session|trial_000_a1_selftest-\d{14}-t000)\.json$/);
    expect(env.logs.join('\n')).toMatch(/Self-test passed/);
  });

  it('uses the STUDY_KEY when one is set', () => {
    const env = ready({ STUDY_KEY: 'k-123' });
    expect(() => env.call('testIngest')).not.toThrow();
  });

  it('throws before setup() so the editor shows the failure', () => {
    const env = load();
    expect(() => env.call('testIngest')).toThrow(/Self-test FAILED/);
  });
});

// ───────────────────────────── the app's view of the replies ─────────────────────────────

describe('contract with the app uploader (apps_script mode)', () => {
  /** fetch double that hands the request body to doPost, like the /exec URL does */
  const viaAppsScript = (env: GasEnv) =>
    vi.fn(async (_url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      expect(init?.headers).toBeUndefined(); // a CORS simple request: Apps Script cannot answer preflights
      const r = env.post(String(init?.body));
      return new Response(r.output.getContent(), { status: 200, headers: { 'Content-Type': 'application/json' } });
    });

  it('stored → ok + confirmed', async () => {
    const env = ready({ STUDY_KEY: 'k-123' });
    const res = await sendEnvelope(appsScriptCfg(), makeEnvelope('trial', makeTrial(), 'k-123'), viaAppsScript(env));
    expect(res).toMatchObject({ ok: true, confirmed: true });
    expect(dataFiles(env)).toHaveLength(1);
  });

  it('busy → a transient failure (the queue retries later)', async () => {
    const env = ready();
    env.locks.contended = true;
    const res = await sendEnvelope(appsScriptCfg({ study_key: '' }), makeEnvelope('session', makeSession()), viaAppsScript(env));
    expect(res).toMatchObject({ ok: false, transient: true });
  });

  it('rejected → a permanent failure for this record only (the queue stops retrying it on its own)', async () => {
    const env = ready({ STUDY_KEY: 'k-123' });
    const res = await sendEnvelope(appsScriptCfg({ study_key: 'wrong' }), makeEnvelope('trial', makeTrial(), 'wrong'), viaAppsScript(env));
    expect(res).toMatchObject({ ok: false, transient: false, permanent: true, error: expect.stringContaining('study key') });
  });

  it('busy and setup-missing answers are never permanent', async () => {
    const busy = ready();
    busy.locks.contended = true;
    expect(await sendEnvelope(appsScriptCfg({ study_key: '' }), makeEnvelope('trial', makeTrial()), viaAppsScript(busy))).toMatchObject({ permanent: false });
    const unconfigured = load();
    expect(await sendEnvelope(appsScriptCfg({ study_key: '' }), makeEnvelope('trial', makeTrial()), viaAppsScript(unconfigured))).toMatchObject({
      ok: false,
      transient: true,
      permanent: false,
    });
  });
});
