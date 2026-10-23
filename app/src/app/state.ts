import type {
  CalibrationResult,
  DeviceInfo,
  Language,
  ParticipantProfile,
  SensorCapabilities,
  SessionRecord,
  StudyConfig,
  UploadConfig,
} from '../core/types';
import { DEFAULT_STUDY_CONFIG } from '../core/constants';

export type ScreenId =
  | 'loading'
  | 'welcome'
  | 'consent'
  | 'profile'
  | 'sensors'
  | 'calibration'
  | 'setup'
  | 'session'
  | 'done'
  | 'data'
  | 'settings'
  | 'sandbox'
  | 'about';

/** Per-device preferences, persisted in localStorage (not research data). */
export interface LocalSettings {
  language: Language;
  audio: boolean;
  speech: boolean;
  vibrate: boolean;
  /** use the built-in sensor simulator instead of real sensors (desktop demos, tests) */
  simulate: boolean;
  /** researcher override of public/config.json upload settings */
  upload_override: Partial<UploadConfig> | null;
  /** id of the participant last used on this device */
  last_participant_id: string | null;
}

export interface OutboxSummary {
  pending: number;
  sent: number;
  failed: number;
}

export interface AppState {
  screen: ScreenId;
  /** where "back" goes from secondary screens (data/settings/sandbox/about) */
  returnTo: ScreenId;
  config: StudyConfig;
  settings: LocalSettings;
  participant: ParticipantProfile | null;
  device: DeviceInfo | null;
  capabilities: SensorCapabilities | null;
  calibration: CalibrationResult | null;
  /** consent accepted in this app run (ISO time); persisted into the session record */
  consentAcceptedAt: string | null;
  session: SessionRecord | null;
  outbox: OutboxSummary;
  toast: { text: string; kind: 'info' | 'ok' | 'warn' | 'error' } | null;
}

const SETTINGS_KEY = 'nosetouch.settings.v1';

export const DEFAULT_SETTINGS: LocalSettings = {
  language: 'en',
  audio: true,
  speech: true,
  vibrate: true,
  simulate: false,
  upload_override: null,
  last_participant_id: null,
};

export function loadSettings(defaultLanguage: Language): LocalSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, language: defaultLanguage, ...JSON.parse(raw), simulate: false };
  } catch {
    // storage blocked (private mode) — fall through to defaults
  }
  return { ...DEFAULT_SETTINGS, language: defaultLanguage };
}

export function persistSettings(s: LocalSettings): void {
  try {
    // `simulate` is deliberately never persisted: a phone left in simulation mode would silently
    // record fake data for the next participant.
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...s, simulate: false }));
  } catch {
    // ignore: settings are a convenience only
  }
}

/**
 * Taps on a control are ignored for this long after it appeared. Consecutive primary buttons sit in
 * the same sticky-footer slot and .btn has touch-action: manipulation, so the second tap of a double
 * tap would otherwise press the NEXT screen's button (e.g. block Start, then Ready: the recording
 * starts before the trial's instruction was ever on screen).
 */
export const TAP_GUARD_MS = 450;

export interface TapGuard {
  /** Call on every render with a key of what is on screen; a new key restarts the guard. */
  shown(key: string): void;
  /** true while a tap would come too soon after the last change of key */
  tooSoon(): boolean;
}

export function createTapGuard(now: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())): TapGuard {
  let current: string | null = null;
  let at = Number.NEGATIVE_INFINITY;
  return {
    shown(key) {
      if (key === current) return;
      current = key;
      at = now();
    },
    tooSoon: () => now() - at < TAP_GUARD_MS,
  };
}

export function initialState(): AppState {
  return {
    screen: 'loading',
    returnTo: 'welcome',
    config: DEFAULT_STUDY_CONFIG,
    settings: { ...DEFAULT_SETTINGS },
    participant: null,
    device: null,
    capabilities: null,
    calibration: null,
    consentAcceptedAt: null,
    session: null,
    outbox: { pending: 0, sent: 0, failed: 0 },
    toast: null,
  };
}
