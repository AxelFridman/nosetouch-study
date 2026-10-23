/**
 * NoseTouch study — upload receiver (Google Apps Script web app, V8 runtime, no libraries).
 *
 * The collector PWA POSTs one UploadEnvelope per record (schema/records.schema.json):
 *   {kind: 'session'|'trial', study_key: string|null, sent_at: ISO time, record: SessionRecord|TrialRecord}
 * as a CORS "simple request" (text/plain body, no custom headers — Apps Script cannot answer CORS
 * preflights). doPost stores ONE Drive file per record and answers JSON through ContentService; Google
 * serves that answer after a 302 to script.googleusercontent.com with Access-Control-Allow-Origin: *,
 * so the page can read it. EVERY path returns ContentService JSON: an uncaught exception would produce
 * an HTML error page without CORS headers, which the browser reports only as "TypeError: Failed to
 * fetch" (and the app would keep retrying without ever learning why).
 *
 * Files (all under the root folder created by setup()):
 *   <study_code>/<participant_id>/<session_id>/session.json
 *   <study_code>/<participant_id>/<session_id>/trial_<trial_index, 3 digits>_a<attempt>_<trial_id>.json
 * Each data file holds exactly ONE record (the SessionRecord / TrialRecord JSON, not the envelope).
 * Re-posting a record overwrites its file (idempotent): the app retries after timeouts — doPost may
 * have stored the record although the answer never arrived — and re-sends the session record after
 * every trial. A rejected upload writes NO Drive file: its body can hold a participant's profile, and
 * a copy outside <study_code>/<participant_id>/ would survive a deletion request.
 *
 * Index spreadsheet (setup() creates it inside the root folder). Every sheet is an append-only log
 * (appending under the lock is cheap and cannot clobber a concurrent write); the Drive files are the
 * data, the sheet is the inventory:
 *   'sessions'   one row per received session upload. The app re-sends the session record after
 *                every trial, so a session appears many times: dedupe by session_id, last row wins.
 *   'trials'     one row per received trial upload; a re-sent trial appears again (dedupe by
 *                trial_id, last row wins).
 *   'quarantine' one row per rejected upload (e.g. a wrong study key = a phone with an old
 *                config.json): when, the reason code, the size, a SHA-256 of the body and the
 *                identifiers found in the envelope — never the body or any profile value. At most
 *                QUARANTINE_DAILY_CAP rows per UTC day; beyond that a rejection writes nothing.
 * String cells are written with a leading apostrophe: Sheets then stores them as plain text (no
 * formula injection from client-supplied text, no ids or versions silently turned into numbers or
 * dates). Stream samples never go into the sheet (50,000 characters per cell).
 *
 * Script Properties: ROOT_FOLDER_ID and INDEX_SPREADSHEET_ID (written by setup()), STUDY_KEY
 * (optional shared token: when set, envelopes must carry it — a spam filter, not real security),
 * MAX_BYTES (request size limit, default 2000000), QUARANTINE_COUNT ('<UTC date>:<rows>', the
 * quarantine rows written that day; written by doPost, delete it to reset today's count).
 *
 * Quotas that shape this code: 6 min per execution and ~30 simultaneous executions per script owner;
 * concurrent appendRow calls lose rows unless they hold the script lock and SpreadsheetApp.flush()
 * before releasing it; Drive file writes are slow, so the lock is never held while a file is written
 * (only while folders are looked up/created — duplicate folders otherwise — and while the index is
 * written). A lock that cannot be obtained in time is answered {ok: false, error: 'busy', retry: true};
 * for a rejection only its quarantine row is skipped (the answer stays retry: false).
 */

const SERVER_VERSION = '1.0.0';
const SERVICE_NAME = 'nosetouch-receiver';
const ROOT_FOLDER_NAME = 'NoseTouchStudy';
const INDEX_SPREADSHEET_NAME = 'NoseTouch index';
const QUARANTINE_SHEET = 'quarantine';
// Reserved because older versions of this script kept rejected payloads in <root>/quarantine/.
const RESERVED_STUDY_CODE = 'quarantine';
const DEFAULT_MAX_BYTES = 2000000;
const QUARANTINE_DAILY_CAP = 500;
const QUARANTINE_COUNT_PROPERTY = 'QUARANTINE_COUNT';
const IDENT_MAX_CHARS = 100;
const CELL_MAX_CHARS = 50000;
const FOLDER_LOCK_MS = 10000;
const INDEX_LOCK_MS = 30000;
const QUARANTINE_LOCK_MS = 5000;
const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
const PARTICIPANT_RE = /^P-[A-HJ-NP-Z2-9]{6}$/;

// ───────────────────────────── index columns ─────────────────────────────

/** [header, value(record, meta)] for a dotted path inside the record. */
function field_(name, path) {
  return [
    name,
    function (record) {
      return pick_(record, path || name);
    },
  ];
}

/** [header, value(record, meta)] for server-side metadata. */
function meta_(name, key) {
  return [
    name,
    function (record, meta) {
      return meta[key || name];
    },
  ];
}

const SESSION_COLUMNS = [
  meta_('received_at'),
  field_('study_code'),
  field_('participant_id'),
  field_('session_id'),
  field_('status'),
  field_('simulated'),
  field_('preset', 'protocol.preset'),
  field_('posture', 'protocol.posture'),
  field_('first_hand', 'protocol.first_hand'),
  field_('n_planned_trials', 'protocol.n_planned_trials'),
  field_('n_attempts', 'progress.n_attempts'),
  field_('n_accepted', 'progress.n_accepted'),
  field_('next_trial_index', 'progress.next_trial_index'),
  field_('started_at'),
  field_('updated_at'),
  field_('ended_at'),
  field_('sex', 'participant.sex'),
  field_('age_years', 'participant.age_years'),
  field_('handedness_self', 'participant.handedness_self'),
  field_('ehi_lq', 'participant.ehi_lq'),
  field_('ehi_class', 'participant.ehi_class'),
  field_('phone_hand', 'participant.phone_hand'),
  field_('platform', 'device.platform'),
  field_('engine', 'device.engine'),
  field_('browser', 'device.browser'),
  field_('os_version', 'device.os_version'),
  field_('model_hint', 'device.model_hint'),
  field_('in_app_browser', 'device.in_app_browser'),
  field_('standalone', 'device.standalone'),
  field_('motion_hz', 'capabilities.measured.motion_hz'),
  field_('permission_state', 'capabilities.permission_state'),
  field_('gravity_sign', 'calibration.gravity_sign'),
  field_('language', 'settings.language'),
  field_('app_version'),
  field_('schema_version'),
  meta_('file_id'),
  meta_('stored_as'),
];

const TRIAL_COLUMNS = [
  meta_('received_at'),
  field_('study_code'),
  field_('participant_id'),
  field_('session_id'),
  field_('trial_id'),
  field_('trial_index'),
  field_('block_index'),
  field_('trial_in_block'),
  field_('attempt'),
  field_('hand', 'condition.hand'),
  field_('orientation', 'condition.orientation'),
  field_('speed', 'condition.speed'),
  field_('start_position', 'condition.start_position'),
  field_('variant', 'condition.variant'),
  field_('posture', 'condition.posture'),
  field_('rep', 'condition.rep'),
  field_('status', 'outcome.status'),
  field_('accepted', 'outcome.accepted'),
  field_('participant_flag', 'outcome.participant_flag'),
  field_('simulated'),
  field_('n_motion', 'quality.n_motion'),
  field_('motion_hz', 'quality.motion_hz'),
  field_('gaps_over_50ms', 'quality.gaps_over_50ms'),
  field_('hidden_during', 'quality.hidden_during'),
  field_('duration_ms', 'segmentation.duration_ms'),
  field_('peak_gyro_dps', 'segmentation.peak_gyro_dps'),
  field_('orientation_match', 'detected.orientation_match'),
  field_('warnings', 'quality.warnings'),
  field_('created_at'),
  field_('app_version'),
  field_('schema_version'),
  meta_('file_id'),
  meta_('stored_as'),
];

/** Metadata and identifiers only (quarantineRow_ fills them in this order). */
const QUARANTINE_COLUMNS = [
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
];

// ───────────────────────────── web app entry points ─────────────────────────────

/** Health check: open the /exec URL in a browser to see it. */
function doGet(e) {
  try {
    const cfg = readConfig_();
    return json_({
      ok: true,
      service: SERVICE_NAME,
      server_version: SERVER_VERSION,
      time: new Date().toISOString(),
      configured: !!(cfg.rootFolderId && cfg.indexSpreadsheetId),
    });
  } catch (err) {
    return json_({
      ok: false,
      service: SERVICE_NAME,
      server_version: SERVER_VERSION,
      time: new Date().toISOString(),
      error: describeError_(err),
    });
  }
}

/** Receives one UploadEnvelope. Never throws: every outcome is a JSON reply. */
function doPost(e) {
  const receivedAt = new Date().toISOString();
  try {
    return json_(handlePost_(e, receivedAt));
  } catch (err) {
    const reply = isBusy_(err)
      ? busyReply_(receivedAt)
      : {
          ok: false,
          error: 'server error: ' + describeError_(err),
          retry: true,
          received_at: receivedAt,
          server_version: SERVER_VERSION,
        };
    try {
      Logger.log('doPost: ' + reply.error);
    } catch (ignored) {
      // logging must never prevent the reply
    }
    return json_(reply);
  }
}

function handlePost_(e, receivedAt) {
  const cfg = readConfig_();
  const post = e && e.postData ? e.postData : null;
  const body = post && typeof post.contents === 'string' ? post.contents : '';
  const bytes = Math.max(body.length, post && typeof post.length === 'number' ? post.length : 0);
  if (!body) return rejection_(cfg, receivedAt, 'empty_body', 'empty request body', '', null, bytes);
  // size check BEFORE parsing: a huge body must not cost a parse
  if (bytes > cfg.maxBytes) {
    const msg = 'payload too large (' + bytes + ' bytes > ' + cfg.maxBytes + ')';
    return rejection_(cfg, receivedAt, 'too_large', msg, body, null, bytes);
  }
  let env;
  try {
    env = JSON.parse(body);
  } catch (err) {
    return rejection_(cfg, receivedAt, 'malformed_json', 'malformed JSON', body, null, bytes);
  }
  const problem = validateEnvelope_(env, cfg.studyKey);
  if (problem) return rejection_(cfg, receivedAt, problem[0], problem[1], body, env, bytes);
  if (!cfg.rootFolderId) {
    // valid data but nowhere to put it: let the app keep it queued until the researcher runs setup()
    return {
      ok: false,
      error: 'receiver not configured: run setup() in the Apps Script editor',
      retry: true,
      received_at: receivedAt,
      server_version: SERVER_VERSION,
    };
  }
  return storeRecord_(cfg, env, receivedAt);
}

// ───────────────────────────── validation ─────────────────────────────

function isObject_(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isId_(v) {
  return typeof v === 'string' && ID_RE.test(v);
}

/** null when the envelope is acceptable, else [reason code, message]. */
function validateEnvelope_(env, studyKey) {
  if (!isObject_(env)) return ['bad_envelope', 'the body must be a JSON object (an UploadEnvelope)'];
  if (env.kind !== 'session' && env.kind !== 'trial') return ['bad_kind', "kind must be 'session' or 'trial'"];
  if (studyKey && env.study_key !== studyKey) return ['bad_study_key', 'study key does not match'];
  const r = env.record;
  if (!isObject_(r)) return ['bad_record', 'record must be an object'];
  if (r.record_type !== env.kind) return ['bad_record_type', 'record.record_type does not match kind'];
  if (typeof r.schema_version !== 'string' || r.schema_version.indexOf('1.') !== 0) {
    return ['bad_schema_version', 'unsupported schema_version (this receiver accepts 1.x)'];
  }
  if (!isId_(r.study_code)) return ['bad_study_code', 'study_code must match ' + ID_RE];
  if (r.study_code.toLowerCase() === RESERVED_STUDY_CODE) return ['bad_study_code', 'study_code is reserved'];
  if (typeof r.participant_id !== 'string' || !PARTICIPANT_RE.test(r.participant_id)) {
    return ['bad_participant_id', 'participant_id must match ' + PARTICIPANT_RE];
  }
  if (!isId_(r.session_id)) return ['bad_session_id', 'session_id must match ' + ID_RE];
  if (env.kind === 'trial') {
    if (!isId_(r.trial_id)) return ['bad_trial_id', 'trial_id must match ' + ID_RE];
    // safe integers only: they print as plain digits in the file name (1e21 would print as '1e+21')
    if (!Number.isSafeInteger(r.trial_index) || r.trial_index < 0) {
      return ['bad_trial_index', 'trial_index must be an integer >= 0'];
    }
    if (!Number.isSafeInteger(r.attempt) || r.attempt < 1) return ['bad_attempt', 'attempt must be an integer >= 1'];
  }
  return null;
}

// ───────────────────────────── storage ─────────────────────────────

function trialFileName_(record) {
  let index = String(record.trial_index);
  while (index.length < 3) index = '0' + index;
  return 'trial_' + index + '_a' + record.attempt + '_' + record.trial_id + '.json';
}

function storeRecord_(cfg, env, receivedAt) {
  const record = env.record;
  const kind = env.kind;
  const root = DriveApp.getFolderById(cfg.rootFolderId);
  const folder = folderPath_(root, [record.study_code, record.participant_id, record.session_id]);
  const name = kind === 'session' ? 'session.json' : trialFileName_(record);
  const file = upsertFile_(folder, name, JSON.stringify(record));
  const storedAs = [record.study_code, record.participant_id, record.session_id, name].join('/');
  const reply = {
    ok: true,
    kind: kind,
    id: kind === 'session' ? record.session_id : record.trial_id,
    stored_as: storedAs,
    received_at: receivedAt,
    server_version: SERVER_VERSION,
  };
  if (!cfg.indexSpreadsheetId) {
    reply.index_error = 'index spreadsheet not configured (run setup())';
    return reply;
  }
  try {
    writeIndex_(cfg.indexSpreadsheetId, kind, record, {
      received_at: receivedAt,
      file_id: file.getId(),
      stored_as: storedAs,
    });
  } catch (err) {
    // the client re-sends on 'busy'; the file write is idempotent and the index row is then added
    if (isBusy_(err)) return busyReply_(receivedAt);
    // The record itself is safely stored: a broken index must not make every phone re-send forever.
    reply.index_error = describeError_(err);
    Logger.log('index write failed for ' + storedAs + ': ' + reply.index_error);
  }
  return reply;
}

/** First non-trashed item of a Drive iterator (DriveApp iterators also list trashed items). */
function firstLive_(iterator) {
  while (iterator.hasNext()) {
    const item = iterator.next();
    if (!item.isTrashed()) return item;
  }
  return null;
}

function childFolder_(parent, name, create) {
  const found = firstLive_(parent.getFoldersByName(name));
  if (found || !create) return found;
  return parent.createFolder(name);
}

/** Gets root/<names...>, creating missing folders under a short script lock (no duplicate folders). */
function folderPath_(root, names) {
  let folder = root;
  for (let i = 0; i < names.length && folder; i++) folder = childFolder_(folder, names[i], false);
  if (folder) return folder; // fast path: the session folder already exists (every trial after the first)
  const lock = acquireLock_(FOLDER_LOCK_MS);
  try {
    folder = root;
    for (let j = 0; j < names.length; j++) folder = childFolder_(folder, names[j], true);
    return folder;
  } finally {
    lock.releaseLock();
  }
}

/** Overwrites the file with this name (idempotent re-posts), or creates it. */
function upsertFile_(folder, name, content) {
  const live = [];
  const it = folder.getFilesByName(name);
  while (it.hasNext()) {
    const f = it.next();
    if (!f.isTrashed()) live.push(f);
  }
  if (live.length === 0) return folder.createFile(name, content, MimeType.PLAIN_TEXT);
  live[0].setContent(content);
  // Two overlapping executions for the same record (a retry while the first was still running) can
  // both create the file; collapse such duplicates whenever the record is posted again.
  for (let i = 1; i < live.length; i++) {
    try {
      live[i].setTrashed(true);
    } catch (err) {
      Logger.log('could not trash duplicate ' + name + ': ' + describeError_(err));
    }
  }
  return live[0];
}

// ───────────────────────────── index spreadsheet ─────────────────────────────

function columnNames_(columns) {
  return columns.map(function (c) {
    return c[0];
  });
}

function ensureSheet_(spreadsheet, name, header) {
  let sheet = spreadsheet.getSheetByName(name);
  if (!sheet) sheet = spreadsheet.insertSheet(name);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(header);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function writeIndex_(spreadsheetId, kind, record, meta) {
  const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
  const columns = kind === 'session' ? SESSION_COLUMNS : TRIAL_COLUMNS;
  const row = columns.map(function (c) {
    return cell_(c[1](record, meta));
  });
  // Concurrent appendRow calls lose rows without the lock; flush() commits before the next
  // execution can take it.
  const lock = acquireLock_(INDEX_LOCK_MS);
  try {
    ensureSheet_(spreadsheet, kind === 'session' ? 'sessions' : 'trials', columnNames_(columns)).appendRow(row);
    SpreadsheetApp.flush();
  } finally {
    lock.releaseLock();
  }
}

/** A sheet cell: numbers/booleans as is, text quote-prefixed (stored literally), null → ''. */
function cell_(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return isFinite(v) ? v : '';
  if (typeof v === 'boolean') return v;
  let s;
  if (Array.isArray(v)) s = v.join('; ');
  else if (typeof v === 'object') s = JSON.stringify(v);
  else s = String(v);
  if (s === '') return '';
  if (s.length > CELL_MAX_CHARS - 1) s = s.slice(0, CELL_MAX_CHARS - 2) + '…';
  return "'" + s;
}

function pick_(obj, path) {
  const parts = path.split('.');
  let v = obj;
  for (let i = 0; i < parts.length; i++) {
    if (v === null || typeof v !== 'object') return null;
    v = v[parts[i]];
  }
  return v === undefined ? null : v;
}

// ───────────────────────────── rejections ─────────────────────────────

/** {ok: false, retry: false}: the record itself was refused. Logs one quarantine row (capped). */
function rejection_(cfg, receivedAt, code, message, body, env, bytes) {
  try {
    quarantine_(cfg, receivedAt, code, body, env, bytes);
  } catch (err) {
    // a busy lock or a Sheets failure costs the row, never the answer: a rejection stays final
    log_('quarantine row not written (' + code + '): ' + describeError_(err));
  }
  return {
    ok: false,
    error: message,
    code: code,
    retry: false,
    received_at: receivedAt,
    server_version: SERVER_VERSION,
  };
}

/**
 * Appends ONE row to the 'quarantine' sheet, so a misconfigured phone can be diagnosed: metadata and
 * the identifiers of the envelope only, never the body (no Drive file either). At most
 * QUARANTINE_DAILY_CAP rows per UTC day, counted in the Script Property QUARANTINE_COUNT
 * ('<UTC date>:<rows>'): beyond the cap a rejection costs one property read and writes nothing, so a
 * phone that keeps an old study key or a spammer cannot fill the sheet. Returns whether a row was written.
 */
function quarantine_(cfg, receivedAt, code, body, env, bytes) {
  if (!cfg.indexSpreadsheetId) return false; // before setup(): nowhere to log
  const props = PropertiesService.getScriptProperties();
  const day = receivedAt.slice(0, 10); // receivedAt is toISOString(): the UTC date
  if (quarantineCount_(props, day) >= QUARANTINE_DAILY_CAP) return false; // no lock, no Sheets call
  const row = quarantineRow_(receivedAt, code, body, env, bytes).map(cell_);
  const spreadsheet = SpreadsheetApp.openById(cfg.indexSpreadsheetId);
  const lock = acquireLock_(QUARANTINE_LOCK_MS);
  try {
    const count = quarantineCount_(props, day); // again under the lock: other executions count too
    if (count >= QUARANTINE_DAILY_CAP) return false;
    ensureSheet_(spreadsheet, QUARANTINE_SHEET, QUARANTINE_COLUMNS).appendRow(row);
    SpreadsheetApp.flush();
    props.setProperty(QUARANTINE_COUNT_PROPERTY, day + ':' + (count + 1));
    return true;
  } finally {
    lock.releaseLock();
  }
}

/** Quarantine rows written on `day` (a UTC 'YYYY-MM-DD'); a count kept for another day is 0 today. */
function quarantineCount_(props, day) {
  const value = String(props.getProperty(QUARANTINE_COUNT_PROPERTY) || '');
  const sep = value.indexOf(':');
  if (sep < 0 || value.slice(0, sep) !== day) return 0;
  const n = parseInt(value.slice(sep + 1), 10);
  return isFinite(n) && n > 0 ? n : 0;
}

/** The QUARANTINE_COLUMNS of one rejected upload. The envelope (null when unparsed) is read best-effort. */
function quarantineRow_(receivedAt, code, body, env, bytes) {
  const e = isObject_(env) ? env : {};
  const r = isObject_(e.record) ? e.record : {};
  return [
    receivedAt,
    code,
    bytes,
    sha256Hex_(body),
    ident_(e.kind),
    ident_(r.record_type),
    ident_(r.study_code),
    ident_(r.participant_id),
    ident_(r.session_id),
    ident_(r.trial_id) || ident_(r.session_id),
    ident_(r.schema_version),
    ident_(r.app_version),
  ];
}

/** An identifier for the quarantine row: a string or finite number, cut to 100 characters; else ''. */
function ident_(v) {
  let s;
  if (typeof v === 'string') s = v;
  else if (typeof v === 'number' && isFinite(v)) s = String(v);
  else return '';
  return s.length > IDENT_MAX_CHARS ? s.slice(0, IDENT_MAX_CHARS) + '…' : s;
}

/** Lower-case hex SHA-256 of the UTF-8 bytes of `text`: tells repeats of one payload apart. */
function sha256Hex_(text) {
  const digest = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, text, Utilities.Charset.UTF_8);
  let hex = '';
  for (let i = 0; i < digest.length; i++) {
    const b = digest[i] & 0xff; // Apps Script bytes are signed (-128..127)
    hex += (b < 16 ? '0' : '') + b.toString(16);
  }
  return hex;
}

// ───────────────────────────── helpers ─────────────────────────────

function readConfig_() {
  const props = PropertiesService.getScriptProperties();
  const maxBytes = parseInt(props.getProperty('MAX_BYTES') || '', 10);
  return {
    rootFolderId: (props.getProperty('ROOT_FOLDER_ID') || '').trim(),
    indexSpreadsheetId: (props.getProperty('INDEX_SPREADSHEET_ID') || '').trim(),
    studyKey: (props.getProperty('STUDY_KEY') || '').trim(),
    maxBytes: isFinite(maxBytes) && maxBytes > 0 ? maxBytes : DEFAULT_MAX_BYTES,
  };
}

function acquireLock_(timeoutMs) {
  const lock = LockService.getScriptLock();
  try {
    lock.waitLock(timeoutMs);
  } catch (err) {
    const busy = new Error('busy: ' + describeError_(err));
    busy.name = 'BusyError';
    throw busy;
  }
  return lock;
}

function isBusy_(err) {
  return !!err && err.name === 'BusyError';
}

function busyReply_(receivedAt) {
  return { ok: false, error: 'busy', retry: true, received_at: receivedAt, server_version: SERVER_VERSION };
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function describeError_(err) {
  const s = err && err.message ? String(err.message) : String(err);
  return s.length > 500 ? s.slice(0, 500) + '…' : s;
}

/** Logger.log that never throws: logging must never change an answer. */
function log_(message) {
  try {
    Logger.log(message);
  } catch (ignored) {
    // nothing else to do
  }
}

function openFolder_(id) {
  if (!id) return null;
  try {
    const folder = DriveApp.getFolderById(id);
    return folder.isTrashed() ? null : folder;
  } catch (err) {
    return null;
  }
}

function openSpreadsheet_(id) {
  if (!id) return null;
  try {
    if (DriveApp.getFileById(id).isTrashed()) return null;
    return SpreadsheetApp.openById(id);
  } catch (err) {
    return null;
  }
}

// ───────────────────────────── run these from the editor ─────────────────────────────

/**
 * One-time setup (safe to run again): creates or reuses the Drive folder 'NoseTouchStudy' and the
 * index spreadsheet inside it, stores their ids in Script Properties and logs their URLs.
 */
function setup() {
  const props = PropertiesService.getScriptProperties();
  let root = openFolder_(props.getProperty('ROOT_FOLDER_ID'));
  if (!root) {
    root = firstLive_(DriveApp.getFoldersByName(ROOT_FOLDER_NAME)) || DriveApp.createFolder(ROOT_FOLDER_NAME);
    props.setProperty('ROOT_FOLDER_ID', root.getId());
  }
  let spreadsheet = openSpreadsheet_(props.getProperty('INDEX_SPREADSHEET_ID'));
  let created = false;
  if (!spreadsheet) {
    const existing = firstLive_(root.getFilesByName(INDEX_SPREADSHEET_NAME));
    if (existing) {
      spreadsheet = SpreadsheetApp.openById(existing.getId());
    } else {
      spreadsheet = SpreadsheetApp.create(INDEX_SPREADSHEET_NAME);
      DriveApp.getFileById(spreadsheet.getId()).moveTo(root);
      created = true;
    }
    props.setProperty('INDEX_SPREADSHEET_ID', spreadsheet.getId());
  }
  ensureSheet_(spreadsheet, 'sessions', columnNames_(SESSION_COLUMNS));
  ensureSheet_(spreadsheet, 'trials', columnNames_(TRIAL_COLUMNS));
  ensureSheet_(spreadsheet, QUARANTINE_SHEET, QUARANTINE_COLUMNS);
  if (created) {
    // drop the empty default sheet ('Sheet1' in English) of a brand-new spreadsheet
    spreadsheet.getSheets().forEach(function (sheet) {
      const name = sheet.getName();
      if (name !== 'sessions' && name !== 'trials' && name !== QUARANTINE_SHEET && sheet.getLastRow() === 0) {
        spreadsheet.deleteSheet(sheet);
      }
    });
  }
  if (!props.getProperty('MAX_BYTES')) props.setProperty('MAX_BYTES', String(DEFAULT_MAX_BYTES));
  Logger.log('Root folder:       ' + root.getUrl());
  Logger.log('Index spreadsheet: ' + spreadsheet.getUrl());
  Logger.log(
    'STUDY_KEY is ' +
      (props.getProperty('STUDY_KEY') ? 'set.' : 'NOT set: any client can upload. Add it in Project Settings > Script Properties.'),
  );
  return { root_folder_id: root.getId(), index_spreadsheet_id: spreadsheet.getId() };
}

/**
 * Self-test: posts a small session + trial through doPost (study_code 'selftest', participant
 * P-TEST22) and logs the replies. Throws when they are not ok. Delete the 'selftest' folder afterwards.
 */
function testIngest() {
  const cfg = readConfig_();
  const now = new Date().toISOString();
  const sessionId = 'selftest-' + now.replace(/[^0-9]/g, '').slice(0, 14);
  const common = {
    schema_version: '1.1.0',
    app_version: 'selftest',
    study_code: 'selftest',
    participant_id: 'P-TEST22',
    session_id: sessionId,
    simulated: true,
  };
  const records = [
    [
      'session',
      Object.assign(
        { record_type: 'session', status: 'completed', started_at: now, updated_at: now, ended_at: now },
        common,
      ),
    ],
    [
      'trial',
      Object.assign(
        {
          record_type: 'trial',
          trial_id: sessionId + '-t000',
          trial_index: 0,
          block_index: 0,
          attempt: 1,
          created_at: now,
          condition: {
            hand: 'right',
            orientation: 'upright_toward',
            speed: 'natural',
            start_position: 'natural',
            variant: 'standard',
            posture: 'standing',
            rep: 0,
          },
          outcome: { status: 'complete', accepted: true, participant_flag: null, note: 'selftest' },
        },
        common,
      ),
    ],
  ];
  const replies = records.map(function (pair) {
    const body = JSON.stringify({ kind: pair[0], study_key: cfg.studyKey || null, sent_at: now, record: pair[1] });
    const reply = JSON.parse(doPost({ postData: { contents: body, length: body.length, type: 'text/plain' } }).getContent());
    Logger.log(pair[0] + ': ' + JSON.stringify(reply));
    return reply;
  });
  const failed = replies.filter(function (r) {
    return r.ok !== true;
  });
  if (failed.length) throw new Error('Self-test FAILED: ' + JSON.stringify(failed));
  Logger.log('Self-test passed. Delete the "selftest" folder in ' + ROOT_FOLDER_NAME + ' when you are done.');
  return replies;
}
