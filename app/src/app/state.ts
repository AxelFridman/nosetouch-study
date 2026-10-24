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

/**
 * Screens. Participant path (spec v2 §1): welcome (with the consent) → profile → session → done.
 * 'sensors' is the repair screen (shown only when the sensors cannot record), 'setup' the researchers'
 * session setup (?setup=1 or Settings); data/settings/sandbox/about are secondary screens.
 */
export type ScreenId =
  | 'loading'
  | 'welcome'
  | 'profile'
  | 'sensors'
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
  /**
   * Researcher setting (lab use): review every trial on a screen with a tap (session settings.review_mode
   * 'tap') instead of the hands-free flow that accepts complete trials automatically ('auto'). Default false.
   */
  review_each_trial: boolean;
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
  /** v2 has no calibration screen: stays null unless a caller sets one (createSession stores it) */
  calibration: CalibrationResult | null;
  /** consent accepted in this app run (ISO time); persisted into the session record */
  consentAcceptedAt: string | null;
  session: SessionRecord | null;
  outbox: OutboxSummary;
  toast: { text: string; kind: 'info' | 'ok' | 'warn' | 'error' } | null;
  /**
   * Researchers: the profile's Continue opens the session setup (presets, custom builder) instead of
   * starting config.default_preset. Set by ?setup=1 or in Settings; like `simulate`, never remembered.
   */
  setupRequested: boolean;
  /**
   * The participant chose "Start over" on the welcome screen although an unfinished session could be
   * resumed: the profile's Continue then starts a new session (the old one ends as abandoned) instead
   * of resuming the recent one. Reset once the session is created.
   */
  startOver: boolean;
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
  review_each_trial: false,
};

export function loadSettings(defaultLanguage: Language): LocalSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const merged: LocalSettings = { ...DEFAULT_SETTINGS, language: defaultLanguage, ...JSON.parse(raw), simulate: false };
      // settings saved before v0.2.0 lack the field; anything but true keeps the hands-free flow
      return { ...merged, review_each_trial: merged.review_each_trial === true };
    }
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
    setupRequested: false,
    startOver: false,
  };
}
