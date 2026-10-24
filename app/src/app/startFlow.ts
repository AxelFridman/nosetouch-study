/**
 * The participant path outside the session (spec v2 §1), as plain functions the screens call:
 *
 *   welcome (+ consent) ──Start──▶ profile ──Continue──▶ session          (the normal path)
 *                                               ├──▶ sensors (repair)    only when the sensors cannot record
 *                                               └──▶ setup               researchers: ?setup=1 or Settings
 *
 * Calibration and setup screens are not on the participant path: Continue creates the session with
 * config.default_preset, posture 'standing' (the session's How-to card can switch to sitting).
 */
import type { PermissionState } from '../core/types';
import { isoTime } from '../core/util';
import { stringsFor } from '../i18n';
import { createSession, recentSessionOf, resumeSession } from './sessionFlow';
import {
  awaitSensorCheck,
  beginSensorCheck,
  needsSensorRepair,
  resetSensorCheck,
  safeRequestPermission,
  SENSOR_WAIT_MS,
  setRepairNext,
  type SensorCheckOutcome,
} from './sensorCheck';
import { getServices } from './services';
import { getState, navigate, setState, showToast, updateSettings } from './store';

export type StartRoute = 'welcome' | 'sensors' | 'setup' | 'session';

/** Where the profile's Continue (and the repair screen's Continue) leads. Pure. */
export function routeAfterProfile(input: {
  /** consent given in this app run (state.consentAcceptedAt) */
  consented: boolean;
  /** ?setup=1 or the Settings switch: the researcher picks the session on the setup screen */
  setupRequested: boolean;
  simulate: boolean;
  secureContext: boolean;
  check: SensorCheckOutcome | null;
}): StartRoute {
  if (!input.consented) return 'welcome';
  if (needsSensorRepair(input)) return 'sensors';
  return input.setupRequested ? 'setup' : 'session';
}

/** ?setup=1 in the page URL (parsed like ?sim=1; never remembered). */
export function setupRequestedIn(search: string): boolean {
  try {
    return new URLSearchParams(search).get('setup') === '1';
  } catch {
    return false;
  }
}

/** First word of a full name ('' when there is none): "Thank you, Dana!". */
export function firstName(fullName: string | null | undefined): string {
  return (fullName ?? '').trim().split(/\s+/)[0] ?? '';
}

/**
 * Inside a tap: unlocks the audio, then keeps the screen on for the session (iOS needs a gesture for
 * both). Never throws.
 */
export function unlockInTap(): void {
  try {
    const services = getServices();
    void services.cues.unlock();
    void services.wakeLock.acquire();
  } catch (e) {
    console.warn('audio / wake lock unavailable', e);
  }
}

/** What the welcome screen's Start tap calls (replaced in tests). */
export interface WelcomeStartDeps {
  /** the motion-permission request (sensorCheck.safeRequestPermission) */
  requestPermission(): Promise<PermissionState>;
  /** audio unlock, then the wake lock (unlockInTap) */
  unlockAudioAndWakeLock(): void;
  /** the consent time (ISO) */
  now(): string;
  /** the background sensor check: measureRates into state.capabilities (sensorCheck.beginSensorCheck) */
  beginSensorCheck(permission: Promise<PermissionState>): unknown;
}

const WELCOME_START_DEPS: WelcomeStartDeps = {
  requestPermission: () => safeRequestPermission(),
  unlockAudioAndWakeLock: () => unlockInTap(),
  now: () => isoTime(),
  beginSensorCheck: (permission) => beginSensorCheck(permission),
};

/**
 * The welcome screen's Start tap (spec v2 §1), synchronous and in this order:
 *  1. the motion-permission request is the FIRST statement: WebKit shows the prompt only while the
 *     gesture is being processed, so nothing (no await, no state update) may come before it;
 *  2. audio unlock and wake lock (iOS needs a gesture for both);
 *  3. the consent time (an earlier one of this app run is kept: back from the profile, Start again);
 *  4. the background sensor check (rates measured into state.capabilities while the profile is filled
 *     in; the profile's Continue waits for it at most SENSOR_WAIT_MS);
 *  5. the profile.
 * Without consent only 1 and 2 happen (the button is disabled then anyway). Returns whether it went on.
 */
export function startFromWelcome(consented: boolean, deps: WelcomeStartDeps = WELCOME_START_DEPS): boolean {
  const permission = deps.requestPermission();
  deps.unlockAudioAndWakeLock();
  if (!consented) return false;
  setState({ consentAcceptedAt: getState().consentAcceptedAt ?? deps.now() });
  void deps.beginSensorCheck(permission);
  navigate('profile');
  return true;
}

/**
 * Continue on the profile (or on the repair screen): waits for the background sensor check (at most
 * SENSOR_WAIT_MS), then opens the repair screen, the researcher's setup screen, or the session: the
 * participant's unfinished session updated within RESUME_ON_CONTINUE_MS (a reload, an app switch,
 * "Exit for now") is resumed where it stopped — unless they chose "Start over" (state.startOver) —
 * else a new one is created with the study's default preset (createSession then ends their older
 * unfinished sessions as abandoned). Resolves with where it went; rejects only when the session could
 * not be created (the caller shows the error).
 */
export async function continueToSession(waitMs: number = SENSOR_WAIT_MS): Promise<StartRoute> {
  const check = await awaitSensorCheck(waitMs);
  const st = getState();
  const route = routeAfterProfile({
    consented: st.consentAcceptedAt !== null,
    setupRequested: st.setupRequested,
    simulate: st.settings.simulate,
    secureContext: st.capabilities?.secure_context ?? (typeof isSecureContext === 'boolean' ? isSecureContext : true),
    check,
  });
  switch (route) {
    case 'welcome':
      navigate('welcome'); // first: navigate() clears the toast
      showToast(stringsFor(st.settings.language).onboarding.setup.needConsent, 'warn');
      break;
    case 'sensors':
      setRepairNext('start');
      navigate('sensors');
      break;
    case 'setup':
      navigate('setup');
      break;
    default: {
      const pid = st.participant?.participant_id;
      const recent = pid && !st.startOver ? await recentSessionOf(pid) : null;
      if (recent && (await resumeSession(recent.session_id).catch(() => null))) {
        navigate('session');
        break;
      }
      await createSession({ preset: st.config.default_preset, posture: 'standing', custom: null });
      setState({ startOver: false });
      navigate('session');
    }
  }
  return route;
}

/**
 * Someone else uses the phone now ("New person" on the welcome screen, "Next person" on the done
 * screen): nothing of the previous person may carry over, and the welcome screen must not greet the
 * next person with the previous one's name (last_participant_id). Their stored profile stays on the
 * phone; they can come back with their participant code.
 */
export function forgetParticipant(): void {
  resetSensorCheck();
  updateSettings({ last_participant_id: null });
  setState({ participant: null, calibration: null, consentAcceptedAt: null, session: null, startOver: false });
}
