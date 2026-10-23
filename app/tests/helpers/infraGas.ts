/**
 * In-memory fakes of the Google Apps Script services that backend/google-apps-script/Code.gs uses,
 * and loadAppsScript(), which runs Code.gs in a node:vm context against them (Apps Script's V8
 * runtime is one shared global scope per project, like a vm context).
 *
 * The fakes copy the documented behaviour that Code.gs depends on:
 *  - DriveApp: getFolderById/getFileById throw for unknown ids; lookups by name ALSO return trashed
 *    items; DriveApp.getFoldersByName searches the whole Drive; iterators throw past the end;
 *    createFile(name, content, mimeType) and setContent refuse content over 10 MB.
 *  - SpreadsheetApp: create() also creates the Drive file (in My Drive, with one sheet 'Sheet1');
 *    appendRow refuses cells over 50,000 characters; a leading apostrophe stores a string as plain
 *    text, a leading '=' would make it a formula; the last sheet cannot be deleted.
 *  - LockService: one script lock; waitLock throws "Lock timeout: ..." while another execution holds
 *    it (locks.contended); a nested acquisition throws (Code.gs never nests locks).
 *  - PropertiesService: string values, getProperty() → null when missing.
 *  - ContentService: TextOutput with getContent/getMimeType; ContentService.MimeType.JSON = 'JSON'.
 * Calls whose order matters (locks, Drive writes, appendRow, flush) are appended to `events`.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

export const CODE_GS_PATH = fileURLToPath(new URL('../../../backend/google-apps-script/Code.gs', import.meta.url));
export const MANIFEST_PATH = fileURLToPath(
  new URL('../../../backend/google-apps-script/appsscript.json', import.meta.url),
);

const TEN_MB = 10 * 1024 * 1024;
export const CELL_LIMIT = 50000;

export const MIME = {
  PLAIN_TEXT: 'text/plain',
  FOLDER: 'application/vnd.google-apps.folder',
  GOOGLE_SHEETS: 'application/vnd.google-apps.spreadsheet',
  CSV: 'text/csv',
  HTML: 'text/html',
  JAVASCRIPT: 'application/javascript',
  PDF: 'application/pdf',
  ZIP: 'application/zip',
} as const;

const CONTENT_MIME = { ATOM: 'ATOM', CSV: 'CSV', ICAL: 'ICAL', JAVASCRIPT: 'JAVASCRIPT', JSON: 'JSON', RSS: 'RSS', TEXT: 'TEXT', VCARD: 'VCARD', XML: 'XML' } as const;

/** A DriveApp FileIterator / FolderIterator. */
export class GasIterator<T> {
  private i = 0;
  constructor(private readonly items: readonly T[]) {}
  hasNext(): boolean {
    return this.i < this.items.length;
  }
  next(): T {
    if (this.i >= this.items.length) throw new Error('Cannot retrieve the next object: iterator has reached the end.');
    return this.items[this.i++];
  }
}

// ───────────────────────────── Drive ─────────────────────────────

export class FakeFile {
  trashed = false;
  setContentCalls = 0;
  constructor(
    private readonly drive: FakeDrive,
    readonly id: string,
    public name: string,
    public parent: FakeFolder | null,
    public content: string,
    public mimeType: string,
  ) {}
  getId(): string {
    return this.id;
  }
  getName(): string {
    return this.name;
  }
  getUrl(): string {
    return `https://drive.google.com/file/d/${this.id}/view`;
  }
  getMimeType(): string {
    return this.mimeType;
  }
  isTrashed(): boolean {
    return this.trashed;
  }
  setTrashed(trashed: boolean): FakeFile {
    this.trashed = !!trashed;
    this.drive.events.push(`drive.setTrashed ${this.path()}`);
    return this;
  }
  setContent(content: string): FakeFile {
    this.drive.checkWritable();
    if (typeof content !== 'string') throw new Error('Invalid argument: content');
    if (Buffer.byteLength(content, 'utf8') > TEN_MB) throw new Error('File content exceeds the 10 MB limit.');
    this.content = content;
    this.setContentCalls++;
    this.drive.events.push(`drive.setContent ${this.path()}`);
    return this;
  }
  getBlob(): { getDataAsString(): string } {
    return { getDataAsString: () => this.content };
  }
  moveTo(destination: FakeFolder): FakeFile {
    if (!(destination instanceof FakeFolder)) throw new Error('Invalid argument: destination');
    this.parent = destination;
    return this;
  }
  path(): string {
    return this.parent ? `${this.parent.path()}/${this.name}` : this.name;
  }
}

export class FakeFolder {
  trashed = false;
  constructor(
    private readonly drive: FakeDrive,
    readonly id: string,
    public name: string,
    public parent: FakeFolder | null,
  ) {}
  getId(): string {
    return this.id;
  }
  getName(): string {
    return this.name;
  }
  getUrl(): string {
    return `https://drive.google.com/drive/folders/${this.id}`;
  }
  isTrashed(): boolean {
    return this.trashed;
  }
  setTrashed(trashed: boolean): FakeFolder {
    this.trashed = !!trashed;
    return this;
  }
  children(): (FakeFolder | FakeFile)[] {
    return [...this.drive.items.values()].filter((item) => item.parent === this);
  }
  subfolders(): FakeFolder[] {
    return this.children().filter((c): c is FakeFolder => c instanceof FakeFolder);
  }
  files(): FakeFile[] {
    return this.children().filter((c): c is FakeFile => c instanceof FakeFile);
  }
  getFoldersByName(name: string): GasIterator<FakeFolder> {
    return new GasIterator(this.subfolders().filter((f) => f.name === name));
  }
  getFilesByName(name: string): GasIterator<FakeFile> {
    return new GasIterator(this.files().filter((f) => f.name === name));
  }
  getFolders(): GasIterator<FakeFolder> {
    return new GasIterator(this.subfolders());
  }
  getFiles(): GasIterator<FakeFile> {
    return new GasIterator(this.files());
  }
  createFolder(name: string): FakeFolder {
    this.drive.checkWritable();
    if (typeof name !== 'string' || name === '') throw new Error('Invalid argument: name');
    const folder = new FakeFolder(this.drive, this.drive.newId('fo'), name, this);
    this.drive.items.set(folder.id, folder);
    this.drive.events.push(`drive.createFolder ${folder.path()}`);
    return folder;
  }
  /** Only the (name, content, mimeType) and (name, content) overloads (Code.gs uses the first). */
  createFile(...args: unknown[]): FakeFile {
    this.drive.checkWritable();
    const [name, content, mimeType] = args;
    if (typeof name !== 'string' || name === '') throw new Error('Invalid argument: name');
    if (typeof content !== 'string') throw new Error('Invalid argument: content');
    if (args.length >= 3 && typeof mimeType !== 'string') throw new Error('Invalid argument: mimeType');
    const limit = args.length >= 3 ? TEN_MB : 50 * 1024 * 1024;
    if (Buffer.byteLength(content, 'utf8') > limit) throw new Error('File content exceeds the size limit.');
    const file = new FakeFile(this.drive, this.drive.newId('fi'), name, this, content, (mimeType as string) ?? MIME.PLAIN_TEXT);
    this.drive.items.set(file.id, file);
    this.drive.events.push(`drive.createFile ${file.path()}`);
    return file;
  }
  path(): string {
    return this.parent ? `${this.parent.path()}/${this.name}` : this.name;
  }
}

export class FakeDrive {
  readonly items = new Map<string, FakeFolder | FakeFile>();
  readonly root: FakeFolder;
  /** set to an Error to make every Drive write throw it (quota, outage) */
  failWrites: Error | null = null;
  private seq = 0;
  constructor(readonly events: string[]) {
    this.root = new FakeFolder(this, 'root', 'My Drive', null);
    this.items.set(this.root.id, this.root);
  }
  newId(prefix: string): string {
    this.seq += 1;
    return `${prefix}${String(this.seq).padStart(4, '0')}`;
  }
  checkWritable(): void {
    if (this.failWrites) throw this.failWrites;
  }
  allFolders(): FakeFolder[] {
    return [...this.items.values()].filter((i): i is FakeFolder => i instanceof FakeFolder && i !== this.root);
  }
  allFiles(): FakeFile[] {
    return [...this.items.values()].filter((i): i is FakeFile => i instanceof FakeFile);
  }
  /** Live (non-trashed) item at a '/'-separated path below `from` (default: My Drive). */
  at(path: string, from: FakeFolder = this.root): FakeFolder | FakeFile | null {
    let node: FakeFolder | FakeFile | null = from;
    for (const part of path.split('/').filter(Boolean)) {
      if (!(node instanceof FakeFolder)) return null;
      const parent: FakeFolder = node;
      node = parent.children().find((c) => c.name === part && !c.trashed) ?? null;
    }
    return node;
  }
  /** The DriveApp global. */
  api() {
    return {
      getFolderById: (id: string): FakeFolder => {
        const item = this.items.get(String(id));
        if (!(item instanceof FakeFolder)) {
          throw new Error('No item with the given ID could be found. Possibly because you have not edited this item or you do not have permission to access it.');
        }
        return item;
      },
      getFileById: (id: string): FakeFile => {
        const item = this.items.get(String(id));
        if (!(item instanceof FakeFile)) {
          throw new Error('No item with the given ID could be found. Possibly because you have not edited this item or you do not have permission to access it.');
        }
        return item;
      },
      getFoldersByName: (name: string) => new GasIterator(this.allFolders().filter((f) => f.name === name)),
      getFilesByName: (name: string) => new GasIterator(this.allFiles().filter((f) => f.name === name)),
      createFolder: (name: string) => this.root.createFolder(name),
      getRootFolder: () => this.root,
    };
  }
}

// ───────────────────────────── Sheets ─────────────────────────────

export interface SheetCell {
  value: string | number | boolean;
  /** written with a leading apostrophe: stored as plain text, never parsed */
  text: boolean;
  /** a string starting with '=': Sheets would evaluate it */
  formula: boolean;
}

function toCell(v: unknown): SheetCell {
  if (v === null || v === undefined) return { value: '', text: false, formula: false };
  if (typeof v === 'number' || typeof v === 'boolean') return { value: v, text: false, formula: false };
  const s = String(v);
  if (s.length > CELL_LIMIT) {
    throw new Error(`Your input contains more than the maximum of ${CELL_LIMIT} characters in a single cell.`);
  }
  if (s.startsWith("'")) return { value: s.slice(1), text: true, formula: false };
  return { value: s, text: false, formula: s.startsWith('=') };
}

export class FakeSheet {
  readonly rows: SheetCell[][] = [];
  frozenRows = 0;
  constructor(
    private readonly events: string[],
    public name: string,
  ) {}
  getName(): string {
    return this.name;
  }
  getLastRow(): number {
    return this.rows.length;
  }
  appendRow(values: unknown[]): FakeSheet {
    if (!Array.isArray(values)) throw new Error('Invalid argument: rowContents');
    const row = values.map(toCell); // validates every cell before anything is stored
    this.rows.push(row);
    this.events.push(`sheet.appendRow ${this.name}`);
    return this;
  }
  setFrozenRows(n: number): FakeSheet {
    this.frozenRows = n;
    return this;
  }
  /** Row values as Sheets would show them (text without the apostrophe). */
  values(): (string | number | boolean)[][] {
    return this.rows.map((r) => r.map((c) => c.value));
  }
  /** Data rows (below the header) as objects keyed by the header. */
  records(): Record<string, string | number | boolean>[] {
    const [header, ...rest] = this.values();
    return rest.map((r) => Object.fromEntries((header ?? []).map((h, i) => [String(h), r[i]])));
  }
}

export class FakeSpreadsheet {
  readonly sheets: FakeSheet[];
  constructor(
    private readonly events: string[],
    readonly id: string,
    public name: string,
  ) {
    this.sheets = [new FakeSheet(events, 'Sheet1')];
  }
  getId(): string {
    return this.id;
  }
  getName(): string {
    return this.name;
  }
  getUrl(): string {
    return `https://docs.google.com/spreadsheets/d/${this.id}/edit`;
  }
  getSheetByName(name: string): FakeSheet | null {
    return this.sheets.find((s) => s.name === name) ?? null;
  }
  insertSheet(name: string): FakeSheet {
    if (this.getSheetByName(name)) throw new Error(`A sheet with the name "${name}" already exists. Please enter another name.`);
    const sheet = new FakeSheet(this.events, name);
    this.sheets.push(sheet);
    this.events.push(`sheets.insertSheet ${name}`);
    return sheet;
  }
  getSheets(): FakeSheet[] {
    return [...this.sheets];
  }
  deleteSheet(sheet: FakeSheet): void {
    if (this.sheets.length <= 1) throw new Error("You can't remove all the sheets in a document.");
    const i = this.sheets.indexOf(sheet);
    if (i < 0) throw new Error('Sheet not found');
    this.sheets.splice(i, 1);
  }
}

export class FakeSpreadsheets {
  readonly byId = new Map<string, FakeSpreadsheet>();
  flushes = 0;
  constructor(
    private readonly drive: FakeDrive,
    private readonly events: string[],
  ) {}
  /** The SpreadsheetApp global. */
  api() {
    return {
      openById: (id: string): FakeSpreadsheet => {
        const ss = this.byId.get(String(id));
        if (!ss || !this.drive.items.has(ss.id)) {
          throw new Error(`Unexpected error while getting the method or property openById on object SpreadsheetApp.`);
        }
        this.events.push('sheets.openById');
        return ss;
      },
      create: (name: string): FakeSpreadsheet => {
        const id = this.drive.newId('ss');
        const ss = new FakeSpreadsheet(this.events, id, name);
        this.byId.set(id, ss);
        this.drive.items.set(id, new FakeFile(this.drive, id, name, this.drive.root, '', MIME.GOOGLE_SHEETS));
        this.events.push(`sheets.create ${name}`);
        return ss;
      },
      flush: (): void => {
        this.flushes++;
        this.events.push('sheets.flush');
      },
    };
  }
}

// ───────────────────────────── Lock / Properties / Content / Logger ─────────────────────────────

export class FakeLocks {
  /** true (or a predicate on the timeout) = another execution holds the script lock */
  contended: boolean | ((timeoutMs: number) => boolean) = false;
  held: object | null = null;
  readonly waits: number[] = [];
  constructor(private readonly events: string[]) {}
  private busy(timeoutMs: number): boolean {
    return typeof this.contended === 'function' ? this.contended(timeoutMs) : this.contended;
  }
  /** The LockService global. */
  api() {
    const scriptLock = () => {
      const lock = {
        waitLock: (timeoutMs: number): void => {
          this.waits.push(timeoutMs);
          this.events.push(`lock.wait ${timeoutMs}`);
          if (this.busy(timeoutMs)) throw new Error('Lock timeout: another process was holding the lock for too long.');
          if (this.held) throw new Error('fake LockService: nested script lock (Code.gs must never nest locks)');
          this.held = lock;
        },
        tryLock: (timeoutMs: number): boolean => {
          this.waits.push(timeoutMs);
          if (this.busy(timeoutMs) || this.held) return false;
          this.held = lock;
          return true;
        },
        hasLock: (): boolean => this.held === lock,
        releaseLock: (): void => {
          if (this.held !== lock) return;
          this.held = null;
          this.events.push('lock.release');
        },
      };
      return lock;
    };
    return {
      getScriptLock: scriptLock,
      getUserLock: scriptLock,
      getDocumentLock: (): null => null, // standalone scripts and web apps get no document lock
    };
  }
}

export class FakeProperties {
  readonly data = new Map<string, string>();
  /** set to an Error to make every read throw it */
  failReads: Error | null = null;
  /** The PropertiesService global. */
  api() {
    const store = {
      getProperty: (key: string): string | null => {
        if (this.failReads) throw this.failReads;
        return this.data.has(key) ? this.data.get(key)! : null;
      },
      setProperty: (key: string, value: unknown) => {
        this.data.set(key, String(value));
        return store;
      },
      deleteProperty: (key: string) => {
        this.data.delete(key);
        return store;
      },
      getProperties: (): Record<string, string> => Object.fromEntries(this.data),
    };
    return { getScriptProperties: () => store };
  }
}

export class FakeTextOutput {
  private mime: string = CONTENT_MIME.TEXT;
  constructor(private content: string) {}
  getContent(): string {
    return this.content;
  }
  setContent(content: string): FakeTextOutput {
    this.content = String(content);
    return this;
  }
  getMimeType(): string {
    return this.mime;
  }
  setMimeType(mime: string): FakeTextOutput {
    if (typeof mime !== 'string' || !(Object.values(CONTENT_MIME) as string[]).includes(mime)) {
      throw new Error('Invalid argument: mimeType');
    }
    this.mime = mime;
    return this;
  }
}

// ───────────────────────────── loader ─────────────────────────────

export interface GasReply {
  output: FakeTextOutput;
  /** the parsed JSON body */
  body: Record<string, unknown>;
}

export interface GasEnv {
  events: string[];
  logs: string[];
  drive: FakeDrive;
  sheets: FakeSpreadsheets;
  locks: FakeLocks;
  props: FakeProperties;
  globals: Record<string, unknown>;
  /** Calls a global function of Code.gs. */
  call<T = unknown>(name: string, ...args: unknown[]): T;
  /** Evaluates an expression in the script's global scope (e.g. a const). */
  evaluate<T = unknown>(expression: string): T;
  /** doPost with a postData body (text/plain unless `type` is given). */
  post(body: string, type?: string): GasReply;
  /** doPost with an arbitrary event object. */
  postEvent(e: unknown): GasReply;
  doGet(): GasReply;
  /** setup(), then the optional Script Properties. */
  setup(props?: Record<string, string>): void;
  /** The study root folder created by setup(). */
  rootFolder(): FakeFolder;
  /** The index spreadsheet created by setup(). */
  index(): FakeSpreadsheet;
}

function asOutput(v: unknown): FakeTextOutput {
  if (!(v instanceof FakeTextOutput)) throw new Error(`expected a ContentService TextOutput, got ${String(v)}`);
  return v;
}

export function loadAppsScript(source: string = readFileSync(CODE_GS_PATH, 'utf8')): GasEnv {
  const events: string[] = [];
  const logs: string[] = [];
  const drive = new FakeDrive(events);
  const sheets = new FakeSpreadsheets(drive, events);
  const locks = new FakeLocks(events);
  const props = new FakeProperties();
  let uuid = 0;
  const logger = {
    log: (...args: unknown[]) => {
      logs.push(args.map(String).join(' '));
      return logger;
    },
  };
  const globals: Record<string, unknown> = {
    DriveApp: drive.api(),
    SpreadsheetApp: sheets.api(),
    LockService: locks.api(),
    PropertiesService: props.api(),
    ContentService: {
      createTextOutput: (content?: string) => new FakeTextOutput(content === undefined ? '' : String(content)),
      MimeType: CONTENT_MIME,
    },
    MimeType: MIME,
    Logger: logger,
    Utilities: {
      getUuid: () => {
        uuid += 1;
        const hex = uuid.toString(16).padStart(12, '0');
        return `00000000-0000-4000-8000-${hex}`;
      },
    },
  };
  const context = vm.createContext(globals);
  vm.runInContext(source, context, { filename: 'Code.gs' });

  const call = <T>(name: string, ...args: unknown[]): T => {
    const fn = (context as Record<string, unknown>)[name];
    if (typeof fn !== 'function') throw new Error(`Code.gs has no global function ${name}()`);
    return fn(...args) as T;
  };
  const postEvent = (e: unknown): GasReply => {
    const output = asOutput(call('doPost', e));
    return { output, body: JSON.parse(output.getContent()) as Record<string, unknown> };
  };
  const env: GasEnv = {
    events,
    logs,
    drive,
    sheets,
    locks,
    props,
    globals,
    call,
    evaluate: <T>(expression: string) => vm.runInContext(expression, context) as T,
    post: (body: string, type = 'text/plain') =>
      postEvent({
        postData: { contents: body, length: Buffer.byteLength(body, 'utf8'), type, name: 'postData' },
        parameter: {},
        parameters: {},
        queryString: '',
        contentLength: Buffer.byteLength(body, 'utf8'),
      }),
    postEvent,
    doGet: () => {
      const output = asOutput(call('doGet', { parameter: {}, parameters: {}, queryString: '' }));
      return { output, body: JSON.parse(output.getContent()) as Record<string, unknown> };
    },
    setup: (extra: Record<string, string> = {}) => {
      call('setup');
      for (const [k, v] of Object.entries(extra)) props.data.set(k, v);
    },
    // direct lookups: these test conveniences must not show up in `events`
    rootFolder: () => {
      const root = drive.items.get(props.data.get('ROOT_FOLDER_ID') ?? '');
      if (!(root instanceof FakeFolder)) throw new Error('no root folder: run env.setup() first');
      return root;
    },
    index: () => {
      const ss = sheets.byId.get(props.data.get('INDEX_SPREADSHEET_ID') ?? '');
      if (!ss) throw new Error('no index spreadsheet: run env.setup() first');
      return ss;
    },
  };
  return env;
}
