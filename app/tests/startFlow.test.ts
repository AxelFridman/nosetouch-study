/**
 * startFlow.continueToSession (the profile's and the repair screen's Continue) with the session
 * creation and the services replaced by fakes: it waits for the background sensor check at most the
 * given time, then opens the repair screen, the researchers' setup screen, or creates the session
 * with config.default_preset, posture 'standing' and opens it. Also forgetParticipant ("New person",
 * "Next person") and unlockInTap.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fakes = vi.hoisted(() => ({
  createSession: vi.fn(async (_opts: unknown) => ({ session_id: 'new' })),
  // the participant's recent unfinished session (none by default) and resuming it
  recentSessionOf: vi.fn(async (_pid: string, _maxAgeMs?: number): Promise<{ session_id: string } | null> => null),
  resumeSession: vi.fn(async (id: string): Promise<{ session_id: string } | null> => ({ session_id: id })),
  services: {
    cues: { unlock: vi.fn(async () => undefined), play: vi.fn(), say: vi.fn(), setOptions: vi.fn() },
    wakeLock: { acquire: vi.fn(async () => true), release: vi.fn(async () => undefined), active: false },
  },
  ready: { value: true },
}));
vi.mock('../src/app/sessionFlow', () => ({
  createSession: fakes.createSession,
  recentSessionOf: fakes.recentSessionOf,
  resumeSession: fakes.resumeSession,
}));
vi.mock('../src/app/services', () => ({
  getServices: () => {
    if (!fakes.ready.value) throw new Error('services not initialised');
    return fakes.services;
  },
}));
const createSession = fakes.createSession;
const fakeServices = fakes.services;

import {
  awaitSensorCheck,
  beginSensorCheck,
  getRepairNext,
  lastSensorCheck,
  resetSensorCheck,
  setRepairNext,
  type SensorCheckDeps,
} from '../src/app/sensorCheck';
import { continueToSession, forgetParticipant, startFromWelcome, unlockInTap, type WelcomeStartDeps } from '../src/app/startFlow';
import { initialState } from '../src/app/state';
import { getState, setState } from '../src/app/store';
import { DEFAULT_STUDY_CONFIG } from '../src/core/constants';
import type { MeasuredRates, PermissionState, SensorCapabilities } from '../src/core/types';
import { makeProfile } from './helpers/dataFixtures';

function rates(hz: number | null): MeasuredRates {
  return {
    duration_ms: 2500,
    motion_hz: hz,
    orientation_hz: hz,
    quaternion_hz: null,
    motion_dt_ms_median: null,
    motion_dt_ms_p95: null,
    acc_available: true,
    accg_available: true,
    rotation_available: true,
    interval_raw: 16,
  };
}

function caps(): SensorCapabilities {
  return {
    secure_context: true,
    device_motion: true,
    device_orientation: true,
    permission_api: true,
    permission_state: 'unknown',
    generic_relative_orientation: false,
    wake_lock: true,
    vibrate: false,
    share_files: false,
    measured: null,
  };
}

/** Runs a background check like the welcome screen's Start tap, with a scripted outcome. */
function check(permission: PermissionState, measure: () => Promise<MeasuredRates>, simulate = false) {
  const deps: SensorCheckDeps = {
    simulate: () => simulate,
    measure,
    patch: (p) => setState({ capabilities: { ...(getState().capabilities ?? caps()), ...p } }),
  };
  return beginSensorCheck(Promise.resolve(permission), deps);
}

beforeEach(() => {
  fakes.ready.value = true;
  createSession.mockClear();
  fakes.recentSessionOf.mockClear();
  fakes.resumeSession.mockClear();
  fakeServices.cues.unlock.mockClear();
  fakeServices.wakeLock.acquire.mockClear();
  const base = initialState();
  setState({
    ...base,
    screen: 'profile',
    config: { ...DEFAULT_STUDY_CONFIG, default_preset: 'street' },
    participant: makeProfile(),
    capabilities: caps(),
    consentAcceptedAt: '2026-10-24T09:00:00.000Z',
  });
});

afterEach(() => resetSensorCheck());

describe('continueToSession', () => {
  it("creates the study's default session, standing, and opens it", async () => {
    await check('granted', async () => rates(60));
    await expect(continueToSession()).resolves.toBe('session');
    expect(createSession).toHaveBeenCalledWith({ preset: 'street', posture: 'standing', custom: null });
    expect(getState().screen).toBe('session');
    expect(getState().capabilities?.measured?.motion_hz).toBe(60); // in the capabilities the session snapshots
  });

  it('uses whatever preset the study config names', async () => {
    setState({ config: { ...DEFAULT_STUDY_CONFIG, default_preset: 'quick' } });
    await continueToSession(0);
    expect(createSession).toHaveBeenCalledWith({ preset: 'quick', posture: 'standing', custom: null });
  });

  it('opens the repair screen when the sensors cannot record, and its Continue then goes on to the session', async () => {
    await check('denied', async () => rates(60));
    setRepairNext('welcome');
    await expect(continueToSession()).resolves.toBe('sensors');
    expect(getState().screen).toBe('sensors');
    expect(getRepairNext()).toBe('start');
    expect(createSession).not.toHaveBeenCalled();

    await check('granted', async () => rates(60)); // fixed on the repair screen: "Measure again"
    await expect(continueToSession()).resolves.toBe('session');
    expect(createSession).toHaveBeenCalledOnce();
  });

  it('a too slow rate also needs the repair screen; the simulator never does', async () => {
    await check('granted', async () => rates(9));
    await expect(continueToSession()).resolves.toBe('sensors');
    setState({ settings: { ...getState().settings, simulate: true } });
    await check('denied', async () => rates(null), true);
    await expect(continueToSession()).resolves.toBe('session');
  });

  it('researchers (?setup=1 / Settings) get the setup screen; no session is created yet', async () => {
    setState({ setupRequested: true });
    await check('granted', async () => rates(60));
    await expect(continueToSession()).resolves.toBe('setup');
    expect(getState().screen).toBe('setup');
    expect(createSession).not.toHaveBeenCalled();
  });

  it('waits for a running measurement at most the given time, then starts anyway', async () => {
    void check('granted', () => new Promise<MeasuredRates>(() => undefined));
    const t0 = Date.now();
    await expect(continueToSession(40)).resolves.toBe('session');
    expect(Date.now() - t0).toBeGreaterThanOrEqual(30);
    expect(lastSensorCheck()).toBeNull();
  });

  it('never starts without consent in this app run: back to the welcome screen', async () => {
    setState({ consentAcceptedAt: null });
    await expect(continueToSession(0)).resolves.toBe('welcome');
    expect(getState().screen).toBe('welcome');
    expect(getState().toast?.kind).toBe('warn');
    expect(createSession).not.toHaveBeenCalled();
  });

  it('a session that cannot be created rejects (the screen shows the error) and stays put', async () => {
    createSession.mockRejectedValueOnce(new Error('disk full'));
    await expect(continueToSession(0)).rejects.toThrow(/disk full/);
    expect(getState().screen).toBe('profile');
  });

  it("resumes the participant's recent unfinished session (a reload, an app switch) instead of starting again", async () => {
    fakes.recentSessionOf.mockResolvedValueOnce({ session_id: 'old' });
    await expect(continueToSession(0)).resolves.toBe('session');
    expect(fakes.recentSessionOf).toHaveBeenCalledWith(makeProfile().participant_id);
    expect(fakes.resumeSession).toHaveBeenCalledWith('old');
    expect(createSession).not.toHaveBeenCalled();
    expect(getState().screen).toBe('session');
  });

  it('"Start over" (state.startOver) starts a new session although one could be resumed; the choice is used once', async () => {
    fakes.recentSessionOf.mockResolvedValue({ session_id: 'old' });
    try {
      setState({ startOver: true });
      await expect(continueToSession(0)).resolves.toBe('session');
      expect(fakes.resumeSession).not.toHaveBeenCalled();
      expect(createSession).toHaveBeenCalledOnce();
      expect(getState().startOver).toBe(false);
    } finally {
      fakes.recentSessionOf.mockReset();
      fakes.recentSessionOf.mockResolvedValue(null);
    }
  });

  it('a recent session that can no longer be resumed (finished meanwhile) → a new one', async () => {
    fakes.recentSessionOf.mockResolvedValueOnce({ session_id: 'old' });
    fakes.resumeSession.mockResolvedValueOnce(null);
    await expect(continueToSession(0)).resolves.toBe('session');
    expect(createSession).toHaveBeenCalledOnce();
  });
});

describe("startFromWelcome (the welcome screen's Start tap)", () => {
  const PERMISSION = Promise.resolve<PermissionState>('granted');
  const AT = '2026-10-24T10:00:00.000Z';

  function scripted(calls: string[]): WelcomeStartDeps {
    return {
      requestPermission: () => {
        calls.push('permission');
        return PERMISSION;
      },
      unlockAudioAndWakeLock: () => void calls.push(`unlock (consent ${getState().consentAcceptedAt === null ? 'not yet' : 'set'})`),
      now: () => {
        calls.push('consent time');
        return AT;
      },
      beginSensorCheck: (p) => void calls.push(`sensor check (${p === PERMISSION ? 'this request' : 'another request'})`),
    };
  }

  beforeEach(() => setState({ screen: 'welcome', consentAcceptedAt: null }));

  it('the permission request FIRST, then audio + wake lock, consent time, background sensor check, profile', () => {
    const calls: string[] = [];
    expect(startFromWelcome(true, scripted(calls))).toBe(true);
    expect(calls).toEqual(['permission', 'unlock (consent not yet)', 'consent time', 'sensor check (this request)']);
    expect(getState()).toMatchObject({ screen: 'profile', consentAcceptedAt: AT });
  });

  it('without consent: only the permission request and the unlock; no consent time, no check, no navigation', () => {
    const calls: string[] = [];
    expect(startFromWelcome(false, scripted(calls))).toBe(false);
    expect(calls).toEqual(['permission', 'unlock (consent not yet)']);
    expect(getState()).toMatchObject({ screen: 'welcome', consentAcceptedAt: null });
  });

  it('keeps the consent time of this app run (back from the profile, then Start again)', () => {
    setState({ consentAcceptedAt: '2026-10-24T09:00:00.000Z' });
    startFromWelcome(true, scripted([]));
    expect(getState().consentAcceptedAt).toBe('2026-10-24T09:00:00.000Z');
  });

  it("with the app's helpers: WebKit's requestPermission() runs synchronously, before the audio unlock and the wake lock", async () => {
    const order: string[] = [];
    const g = globalThis as { DeviceMotionEvent?: unknown };
    g.DeviceMotionEvent = {
      requestPermission: () => {
        order.push('DeviceMotionEvent.requestPermission');
        return Promise.resolve('granted');
      },
    };
    fakeServices.cues.unlock.mockImplementationOnce(async () => void order.push('cues.unlock'));
    fakeServices.wakeLock.acquire.mockImplementationOnce(async () => {
      order.push('wakeLock.acquire');
      return true;
    });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined); // the fake services cannot measure
    try {
      expect(startFromWelcome(true)).toBe(true);
      // all of it happened inside the tap, before any await
      expect(order).toEqual(['DeviceMotionEvent.requestPermission', 'cues.unlock', 'wakeLock.acquire']);
      expect(getState().screen).toBe('profile');
      expect(getState().consentAcceptedAt).not.toBeNull();
      // the background check got this tap's answer and stored it for the session record
      expect((await awaitSensorCheck(1000))?.permission).toBe('granted');
      expect(getState().capabilities?.permission_state).toBe('granted');
    } finally {
      delete g.DeviceMotionEvent;
      warn.mockRestore();
    }
  });
});

describe('forgetParticipant ("New person", "Next person")', () => {
  it('clears the participant, the consent, the session and the sensor check', async () => {
    await check('granted', async () => rates(60));
    setState({ session: { session_id: 's' } as never, settings: { ...getState().settings, last_participant_id: 'P-ABCDEF' } });
    forgetParticipant();
    const st = getState();
    expect(st.participant).toBeNull();
    expect(st.consentAcceptedAt).toBeNull();
    expect(st.session).toBeNull();
    expect(st.calibration).toBeNull();
    expect(st.settings.last_participant_id).toBeNull(); // the welcome screen never greets the next person by name
    expect(lastSensorCheck()).toBeNull();
  });
});

describe('unlockInTap', () => {
  it('unlocks audio and takes the wake lock; never throws without services', () => {
    unlockInTap();
    expect(fakeServices.cues.unlock).toHaveBeenCalledOnce();
    expect(fakeServices.wakeLock.acquire).toHaveBeenCalledOnce();
    fakes.ready.value = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(() => unlockInTap()).not.toThrow();
    warn.mockRestore();
  });
});
