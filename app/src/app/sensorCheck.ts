/**
 * Motion permission and the sensor check, without UI (the repair screen SensorCheckScreen.tsx shows it).
 *
 * v2 participant path (spec §1): the welcome screen's Start tap requests the motion permission as its
 * FIRST synchronous statement (WebKit only shows the prompt while the gesture is being processed) and
 * starts beginSensorCheck(): once the permission is answered, the real sampling rates are measured for
 * MEASURE_MS in the background, into state.capabilities (snapshotted into the session record), while
 * the participant fills in the profile. The profile's Continue waits for the result at most
 * SENSOR_WAIT_MS (awaitSensorCheck) and opens the repair screen only when the sensors cannot record
 * (needsSensorRepair). The latest started check always wins: an older run that settles later never
 * overwrites a newer result.
 */
import { detectCapabilities, measureRates, requestMotionPermission } from '../core/sensors';
import type { MeasuredRates, PermissionState, SensorCapabilities } from '../core/types';
import { getServices } from './services';
import { getState, setState } from './store';

/** Length of the rate measurement. */
export const MEASURE_MS = 2500;
/** Below this the movement cannot be recorded usefully: continuing is blocked. */
export const MIN_MOTION_HZ = 20;
/** Typical browsers deliver ~60 Hz; below this we warn (Low Power Mode, throttling). */
export const GOOD_MOTION_HZ = 45;
/** How long the profile's Continue waits for the background measurement before it starts anyway. */
export const SENSOR_WAIT_MS = 3000;

/**
 * Starts the motion-permission request. Call it as the FIRST statement of a tap handler, before any
 * await: WebKit only shows the prompt while the gesture is being processed. Never throws/rejects
 * ('unknown' when the request itself failed).
 */
export function safeRequestPermission(): Promise<PermissionState> {
  try {
    return requestMotionPermission().catch((e: unknown) => {
      console.warn('motion permission request failed', e);
      return 'unknown' as const;
    });
  } catch (e) {
    console.warn('motion permission request failed', e);
    return Promise.resolve('unknown');
  }
}

/** Capabilities when feature detection itself failed: conservative, but schema-valid. */
export function fallbackCapabilities(): SensorCapabilities {
  return {
    secure_context: typeof isSecureContext === 'boolean' ? isSecureContext : false,
    device_motion: false,
    device_orientation: false,
    permission_api: false,
    permission_state: 'unknown',
    generic_relative_orientation: false,
    wake_lock: false,
    vibrate: false,
    share_files: false,
    measured: null,
  };
}

export function currentCapabilities(): SensorCapabilities {
  const caps = getState().capabilities;
  if (caps) return caps;
  try {
    return detectCapabilities();
  } catch (e) {
    console.warn('detectCapabilities failed', e);
    return fallbackCapabilities();
  }
}

export function patchCapabilities(patch: Partial<SensorCapabilities>): void {
  setState({ capabilities: { ...currentCapabilities(), ...patch } });
}

export type SensorIssue = 'insecure' | 'denied' | 'failed' | 'no_events' | 'too_slow' | 'slow' | 'no_gyro';

export interface SensorVerdict {
  canContinue: boolean;
  issues: SensorIssue[];
  /** rounded motion rate, or null when nothing was measured */
  hz: number | null;
}

/** What a measurement found, independent of the UI (unit-tested). */
export function sensorVerdict(input: {
  simulate: boolean;
  secureContext: boolean;
  permission: PermissionState | null;
  measured: MeasuredRates | null;
  failed: boolean;
}): SensorVerdict {
  const raw = input.measured?.motion_hz ?? null;
  const hz = raw !== null && Number.isFinite(raw) ? Math.round(raw) : null;
  // The simulator delivers data regardless of the browser, so nothing can block the flow.
  if (input.simulate) return { canContinue: true, issues: [], hz };
  const issues: SensorIssue[] = [];
  if (!input.secureContext) issues.push('insecure');
  if (input.permission === 'denied') issues.push('denied');
  if (input.failed) issues.push('failed');
  if (input.measured) {
    if (hz === null || hz <= 0) issues.push('no_events');
    else if (hz < MIN_MOTION_HZ) issues.push('too_slow');
    else if (hz < GOOD_MOTION_HZ) issues.push('slow');
    if (hz !== null && hz > 0 && !input.measured.rotation_available) issues.push('no_gyro');
  }
  const canContinue = input.permission !== 'denied' && raw !== null && raw >= MIN_MOTION_HZ;
  return { canContinue, issues, hz };
}

// ───────────────────────────── background check (welcome Start → profile Continue) ─────────────────────────────

export interface SensorCheckOutcome {
  permission: PermissionState;
  /** null when nothing was measured (permission denied, or the measurement failed) */
  measured: MeasuredRates | null;
  /** the measurement itself threw */
  failed: boolean;
}

export interface SensorCheckDeps {
  /** simulation mode: the simulator needs no permission */
  simulate(): boolean;
  /** measures the sampling rates of a fresh sensor source for `ms` */
  measure(ms: number): Promise<MeasuredRates>;
  /** stores permission/measurement into state.capabilities */
  patch(p: Partial<SensorCapabilities>): void;
}

function appDeps(): SensorCheckDeps {
  return {
    simulate: () => getState().settings.simulate,
    measure: (ms) => measureRates(getServices().createSource(), ms),
    patch: patchCapabilities,
  };
}

interface Run {
  id: number;
  promise: Promise<SensorCheckOutcome>;
  outcome: SensorCheckOutcome | null;
}

let current: Run | null = null;
let runSeq = 0;

/**
 * Waits for the permission answer, then measures the rates (unless denied). Patches the capabilities
 * only while `isCurrent()` holds, so an older run never overwrites a newer one. Never rejects.
 */
export async function runSensorCheck(
  permission: Promise<PermissionState>,
  deps: SensorCheckDeps,
  isCurrent: () => boolean = () => true,
): Promise<SensorCheckOutcome> {
  let perm: PermissionState;
  try {
    perm = await permission;
  } catch {
    perm = 'unknown';
  }
  if (deps.simulate()) perm = 'granted';
  if (isCurrent()) deps.patch({ permission_state: perm, measured: null });
  if (perm === 'denied') return { permission: perm, measured: null, failed: false };
  try {
    const measured = await deps.measure(MEASURE_MS);
    if (isCurrent()) deps.patch({ permission_state: perm, measured });
    return { permission: perm, measured, failed: false };
  } catch (e) {
    console.warn('sensor measurement failed', e);
    return { permission: perm, measured: null, failed: true };
  }
}

/**
 * Starts a background check with the permission request of the current tap (see the module doc).
 * Replaces any earlier check. Returns its promise (never rejects).
 */
export function beginSensorCheck(permission: Promise<PermissionState>, deps: SensorCheckDeps = appDeps()): Promise<SensorCheckOutcome> {
  const id = ++runSeq;
  const isCurrent = () => current?.id === id;
  // runSensorCheck awaits the permission first, so `current` below is set before isCurrent() runs
  const promise = runSensorCheck(permission, deps, isCurrent).then((outcome) => {
    if (current?.id === id) current.outcome = outcome;
    return outcome;
  });
  current = { id, promise, outcome: null };
  return promise;
}

/** The latest check's outcome, waiting at most `maxWaitMs`; null when none was started or it is still running. */
export function awaitSensorCheck(maxWaitMs: number = SENSOR_WAIT_MS): Promise<SensorCheckOutcome | null> {
  const run = current;
  if (!run) return Promise.resolve(null);
  if (run.outcome) return Promise.resolve(run.outcome);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), Math.max(0, maxWaitMs));
    run.promise.then(
      (outcome) => {
        clearTimeout(timer);
        resolve(outcome);
      },
      () => {
        clearTimeout(timer);
        resolve(null);
      },
    );
  });
}

/** The latest check's outcome if it has finished (no waiting). */
export function lastSensorCheck(): SensorCheckOutcome | null {
  return current?.outcome ?? null;
}

// ───────────────────────────── where the repair screen's Continue leads ─────────────────────────────

/** 'start': on to the session (the normal path); 'welcome': back to the resume card. */
export type RepairNext = 'start' | 'welcome';

let repairNext: RepairNext = 'start';

/** Remembers where the repair screen's Continue goes; set by whoever opens the repair screen. */
export function setRepairNext(next: RepairNext): void {
  repairNext = next;
}

export function getRepairNext(): RepairNext {
  return repairNext;
}

/** Forgets the check (a new participant, or the sensor source changed between simulator and real sensors). */
export function resetSensorCheck(): void {
  current = null;
  repairNext = 'start';
}

/**
 * true when the participant must see the repair screen before a session can record: permission
 * denied, no secure context, the measurement failed, no motion events, or fewer than MIN_MOTION_HZ.
 * A check that has not finished (null) never blocks: the trials detect a sensor stall themselves.
 */
export function needsSensorRepair(input: { simulate: boolean; secureContext: boolean; check: SensorCheckOutcome | null }): boolean {
  if (input.simulate) return false;
  if (!input.secureContext) return true;
  const c = input.check;
  if (!c) return false;
  if (c.permission === 'denied' || c.failed) return true;
  if (!c.measured) return false;
  const hz = c.measured.motion_hz;
  return hz === null || !Number.isFinite(hz) || hz < MIN_MOTION_HZ;
}
