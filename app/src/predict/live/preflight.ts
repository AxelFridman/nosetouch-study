/**
 * What the page knows before the first tap, without prompting (types.ts Preflight): core/sensors.ts
 * detectCapabilities() and getDeviceInfo(). A failure here blocks the Start button and shows the
 * 'unsupported' screen; the demo (&sim=1) ignores it.
 */
import { detectCapabilities, getDeviceInfo } from '../../core/sensors';
import type { DeviceInfo, SensorCapabilities } from '../../core/types';
import type { BrowserEngine, Preflight, SourceFailure } from '../types';

export function preflightFrom(caps: SensorCapabilities, device: Pick<DeviceInfo, 'platform' | 'engine' | 'in_app_browser'>): Preflight {
  let failure: SourceFailure | null = null;
  if (!caps.secure_context) failure = 'insecure_context';
  else if (device.in_app_browser) failure = 'in_app_browser';
  else if (!caps.device_motion) failure = 'not_supported';
  const engine: BrowserEngine = device.engine ?? 'unknown';
  return {
    ok: failure === null,
    failure,
    engine,
    needsPermission: caps.permission_api,
    permission: caps.permission_state,
    wakeLock: caps.wake_lock,
    vibrate: caps.vibrate,
    desktop: device.platform === 'other',
  };
}

export async function runPreflight(): Promise<Preflight> {
  const caps = detectCapabilities();
  let device: Pick<DeviceInfo, 'platform' | 'engine' | 'in_app_browser'>;
  try {
    device = await getDeviceInfo();
  } catch {
    device = { platform: 'other', engine: 'unknown', in_app_browser: null };
  }
  return preflightFrom(caps, device);
}

/** The same flags for the demo: nothing blocks it and it never prompts, vibrates or keeps the screen on. */
export function demoPreflight(engine: BrowserEngine): Preflight {
  return {
    ok: true,
    failure: null,
    engine,
    needsPermission: false,
    permission: 'not_required',
    wakeLock: false,
    vibrate: false,
    desktop: false,
  };
}
