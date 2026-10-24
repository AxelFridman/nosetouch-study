/**
 * Study configuration (public/config.json, editable after deployment without rebuilding) and
 * privacy redaction.
 *
 * loadStudyConfig(): fetch('./config.json', {cache: 'no-store'}); deep-merge over
 * DEFAULT_STUDY_CONFIG; sanitize (study_code must match [A-Za-z0-9_-]{1,64} else default;
 * upload.mode must be one of the UploadMode values else 'none'; endpoint must be https:// or a
 * same-origin relative path like '/api/ingest' else ''); never throws.
 * The fetch is abandoned after 8 s: boot awaits this, and a hanging request (captive portal, dead
 * Wi-Fi) must not leave the participant staring at a spinner. Every other field is type-checked
 * individually; a bad field falls back to its default without discarding the rest.
 * Offline: every successfully loaded config is remembered in localStorage (CONFIG_CACHE_KEY, tagged
 * with the config URL because all GitHub Pages sites of one user share an origin). When the fetch
 * fails (network error, timeout, HTTP error, a body that is not a JSON object such as a captive
 * portal's page), that copy is used, and the built-in defaults only when there is none: the
 * defaults carry the wrong study_code / consent_version and switch uploads off.
 * fetchStudyConfig() also says where the config came from; refetchConfigWhenOnline() lets the app
 * replace a fallback with the deployed config once the phone is back online.
 * effectiveUpload(): config.upload overlaid with a non-empty LocalSettings.upload_override (field by
 * field: empty/invalid override fields are ignored, so a half-filled settings form cannot blank a
 * working endpoint).
 * redactProfile(): returns a copy with nickname = null unless cfg.upload_nickname.
 * redactSession(): same for session.participant.
 * DOM-free (fetch/AbortController are injected or feature-checked; both exist in Node 24).
 *
 * IMPLEMENTED BY: data agent.
 */
import { DEFAULT_STUDY_CONFIG } from './constants';
import type {
  Language,
  LocalizedText,
  ParticipantProfile,
  ProtocolPreset,
  SessionRecord,
  StudyConfig,
  UploadConfig,
  UploadMode,
} from './types';

export const CONFIG_URL = './config.json';
export const CONFIG_TIMEOUT_MS = 8000;

const STUDY_CODE_RE = /^[A-Za-z0-9_-]{1,64}$/;
const UPLOAD_MODES: readonly UploadMode[] = ['apps_script', 'json_post', 'none'];
const LANGUAGES: readonly Language[] = ['en', 'he'];
/** Every ProtocolPreset (a config naming another value falls back to the default preset). */
export const PRESETS: readonly ProtocolPreset[] = ['street', 'quick', 'standard', 'full', 'custom'];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function oneOf<T extends string>(values: readonly T[], v: unknown): v is T {
  return typeof v === 'string' && (values as readonly string[]).includes(v);
}

/** Trimmed string if `v` is a string of at most `max` characters after trimming, else undefined. */
function cleanString(v: unknown, max: number): string | undefined {
  if (typeof v !== 'string') return undefined;
  const s = v.trim();
  return s.length <= max ? s : undefined;
}

/**
 * Upload endpoint check: an absolute https:// URL (no embedded credentials: fetch rejects them), or
 * a same-origin relative path ('/api/ingest', './ingest'). Protocol-relative '//host' and
 * backslashes (browsers read '/\host' as '//host') would leave the origin, so they are refused.
 */
export function isValidEndpoint(endpoint: string): boolean {
  const v = endpoint.trim();
  if (v === '' || /[\s\\]/.test(v) || /[\u0000-\u001f\u007f]/.test(v)) return false;
  if (v.startsWith('/')) return !v.startsWith('//');
  if (v.startsWith('./') || v.startsWith('../')) return true;
  try {
    const u = new URL(v);
    return u.protocol === 'https:' && u.hostname !== '' && u.username === '' && u.password === '';
  } catch {
    return false;
  }
}

function defaults(): StudyConfig {
  const d = DEFAULT_STUDY_CONFIG;
  return {
    ...d,
    upload: { ...d.upload },
    contact: { ...d.contact },
    institution: { ...d.institution },
    region: { ...d.region },
  };
}

/** {en, he} object, or one plain string used for both languages; anything else → empty texts. */
function cleanLocalized(raw: unknown, max: number): LocalizedText {
  if (typeof raw === 'string') {
    const s = cleanString(raw, max) ?? '';
    return { en: s, he: s };
  }
  const r = isRecord(raw) ? raw : {};
  return { en: cleanString(r.en, max) ?? '', he: cleanString(r.he, max) ?? '' };
}

/** The text for `lang`, falling back to English, then to ''. */
export function localizedText(text: LocalizedText | undefined, lang: Language): string {
  return text?.[lang] || text?.en || '';
}

export function sanitizeStudyConfig(raw: unknown): StudyConfig {
  const d = defaults();
  const r = isRecord(raw) ? raw : {};
  const up = isRecord(r.upload) ? r.upload : {};
  const contact = isRecord(r.contact) ? r.contact : {};

  const studyCode = cleanString(r.study_code, 64);
  const endpoint = typeof up.endpoint === 'string' && isValidEndpoint(up.endpoint) ? up.endpoint.trim() : '';
  const minAge = r.min_age;

  return {
    study_code: studyCode !== undefined && STUDY_CODE_RE.test(studyCode) ? studyCode : d.study_code,
    upload: {
      mode: oneOf(UPLOAD_MODES, up.mode) ? up.mode : 'none',
      endpoint,
      study_key: cleanString(up.study_key, 200) ?? d.upload.study_key,
    },
    upload_nickname: typeof r.upload_nickname === 'boolean' ? r.upload_nickname : d.upload_nickname,
    consent_version: cleanString(r.consent_version, 200) || d.consent_version,
    contact: {
      name: cleanString(contact.name, 200) ?? d.contact.name,
      email: cleanString(contact.email, 200) ?? d.contact.email,
    },
    institution: cleanLocalized(r.institution, 120),
    region: cleanLocalized(r.region, 80),
    default_language: oneOf(LANGUAGES, r.default_language) ? r.default_language : d.default_language,
    default_preset: oneOf(PRESETS, r.default_preset) ? r.default_preset : d.default_preset,
    min_age:
      typeof minAge === 'number' && Number.isFinite(minAge) && minAge >= 0 && minAge <= 120
        ? Math.round(minAge)
        : d.min_age,
  };
}

/** localStorage key of the last config.json that loaded successfully (see the module doc). */
export const CONFIG_CACHE_KEY = 'nosetouch.config.v1';

/** Where a loaded config came from: the server (or the service worker's copy), this phone's last good copy, or the built-in defaults. */
export type ConfigSource = 'network' | 'cache' | 'defaults';

export interface LoadedConfig {
  config: StudyConfig;
  source: ConfigSource;
}

type ConfigStorage = Pick<Storage, 'getItem' | 'setItem'>;

/** window.localStorage, or null where it is missing or blocked (accessing it can throw). */
function defaultStorage(): ConfigStorage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
}

/** The absolute URL of config.json: the remembered copy is only valid for the same deployment. */
function configScope(): string {
  try {
    const base = typeof document !== 'undefined' ? document.baseURI : globalThis.location?.href;
    return base ? new URL(CONFIG_URL, base).href : CONFIG_URL;
  } catch {
    return CONFIG_URL;
  }
}

function rememberConfig(storage: ConfigStorage | null, config: StudyConfig): void {
  try {
    storage?.setItem(CONFIG_CACHE_KEY, JSON.stringify({ url: configScope(), saved_at: new Date().toISOString(), config }));
  } catch {
    // storage full or blocked: the next offline start falls back to the defaults
  }
}

function rememberedConfig(storage: ConfigStorage | null): StudyConfig | null {
  try {
    const raw = storage?.getItem(CONFIG_CACHE_KEY);
    if (!raw) return null;
    const entry: unknown = JSON.parse(raw);
    if (!isRecord(entry) || entry.url !== configScope() || !isRecord(entry.config)) return null;
    return sanitizeStudyConfig(entry.config);
  } catch {
    return null;
  }
}

/**
 * Loads config.json (see the module doc) and reports where the result came from. Never throws.
 * `storage` defaults to localStorage; pass null to neither read nor write a remembered copy.
 */
export async function fetchStudyConfig(fetchImpl?: typeof fetch, storage: ConfigStorage | null = defaultStorage()): Promise<LoadedConfig> {
  const doFetch = fetchImpl ?? (typeof fetch === 'function' ? fetch : undefined);
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => {
      ctrl?.abort();
      resolve(null);
    }, CONFIG_TIMEOUT_MS);
  });
  try {
    if (!doFetch) throw new Error('fetch is not available');
    const init: RequestInit = { cache: 'no-store' };
    if (ctrl) init.signal = ctrl.signal;
    const res = await Promise.race([doFetch(CONFIG_URL, init), timeout]);
    if (!res) throw new Error(`timed out after ${CONFIG_TIMEOUT_MS} ms`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await Promise.race([res.text(), timeout]);
    if (text === null) throw new Error(`timed out after ${CONFIG_TIMEOUT_MS} ms`);
    const raw: unknown = JSON.parse(text);
    if (!isRecord(raw)) throw new Error('config.json is not a JSON object');
    const config = sanitizeStudyConfig(raw);
    rememberConfig(storage, config);
    return { config, source: 'network' };
  } catch (e) {
    const cached = rememberedConfig(storage);
    console.warn(`config.json could not be loaded; using ${cached ? 'the copy saved on this phone' : 'the built-in defaults'}`, e);
    return cached ? { config: cached, source: 'cache' } : { config: defaults(), source: 'defaults' };
  } finally {
    clearTimeout(timer);
  }
}

/** The study configuration (see fetchStudyConfig). Never throws. */
export async function loadStudyConfig(fetchImpl?: typeof fetch, storage?: ConfigStorage | null): Promise<StudyConfig> {
  return (await fetchStudyConfig(fetchImpl, storage)).config;
}

/**
 * After a start on a fallback config: loads config.json again at every 'online' event of `target`
 * until the network answers, then hands the fresh config to `apply` once and stops listening.
 * Returns a function that stops listening.
 */
export function refetchConfigWhenOnline(
  target: Pick<EventTarget, 'addEventListener' | 'removeEventListener'>,
  apply: (config: StudyConfig) => void,
  load: () => Promise<LoadedConfig> = () => fetchStudyConfig(),
): () => void {
  let busy = false;
  let stopped = false;
  const stop = () => {
    stopped = true;
    target.removeEventListener('online', onOnline);
  };
  function onOnline(): void {
    if (busy || stopped) return;
    busy = true;
    load()
      .then((loaded) => {
        if (stopped || loaded.source !== 'network') return;
        stop();
        apply(loaded.config);
      })
      .catch((e: unknown) => console.warn('reloading config.json failed', e))
      .finally(() => {
        busy = false;
      });
  }
  target.addEventListener('online', onOnline);
  return stop;
}

export function effectiveUpload(cfg: StudyConfig, override: Partial<UploadConfig> | null): UploadConfig {
  const out: UploadConfig = { ...cfg.upload };
  if (!isRecord(override)) return out;
  if (oneOf(UPLOAD_MODES, override.mode)) out.mode = override.mode;
  if (typeof override.endpoint === 'string' && isValidEndpoint(override.endpoint)) {
    out.endpoint = override.endpoint.trim();
  }
  if (typeof override.study_key === 'string' && override.study_key.trim() !== '') {
    out.study_key = override.study_key.trim();
  }
  return out;
}

export function redactProfile(p: ParticipantProfile, cfg: StudyConfig): ParticipantProfile {
  return {
    ...p,
    nickname: cfg.upload_nickname ? p.nickname : null,
    ehi_sf: p.ehi_sf ? { ...p.ehi_sf } : null,
  };
}

export function redactSession(s: SessionRecord, cfg: StudyConfig): SessionRecord {
  return { ...s, participant: redactProfile(s.participant, cfg) };
}
