/**
 * LiveSource over the real sensors: core/sensors.ts createBrowserSensorSource (capture, engine handling)
 * with only devicemotion (no quaternion, absolute orientation, magnetometer or screen events), plus the
 * first-second checks of docs/stage3/APP_SPEC.md section 4.2 (no_gyroscope, no_accelerometer).
 *
 * The permission prompt is NOT here: core/sensors.ts requestMotionPermission() must be the first
 * statement of the Start tap (iOS shows the prompt only inside the user gesture), so PredictApp calls it
 * and start() never prompts. Values are delivered RAW (no sign fix, no rounding): the engine's LiveBuffer
 * rounds like the recorder and the pipeline applies the engine's gravity sign.
 */
import { createBrowserSensorSource } from '../../core/sensors';
import type { MotionSample, SensorError } from '../../core/types';
import type { BrowserEngine, LiveSample, LiveSource, SourceFailure } from '../types';

/** the window of the gyro / accelerometer checks (ms of samples) */
export const SENSOR_CHECK_MS = 1000;

export function failureOfSensorError(code: SensorError['code']): SourceFailure {
  switch (code) {
    case 'insecure_context':
      return 'insecure_context';
    case 'not_supported':
      return 'not_supported';
    case 'permission_denied':
      return 'permission_denied';
    case 'no_events':
    case 'sensor_error':
    default:
      return 'no_events';
  }
}

/**
 * The first-second check: the reason to stop, or null. Every sample of the first SENSOR_CHECK_MS without
 * accelerationIncludingGravity -> no_accelerometer; without rotationRate -> no_gyroscope.
 */
export function createSensorCheck(): (s: MotionSample) => SourceFailure | null | 'ok' {
  let t0: number | null = null;
  let accg = false;
  let gyro = false;
  let done = false;
  return (s) => {
    if (done) return 'ok';
    if (t0 === null) t0 = s.t;
    if (s.accg) accg = true;
    if (s.rr) gyro = true;
    if (s.t - t0 < SENSOR_CHECK_MS) return null;
    done = true;
    if (!accg) return 'no_accelerometer';
    if (!gyro) return 'no_gyroscope';
    return 'ok';
  };
}

export function createBrowserLiveSource(engine: BrowserEngine): LiveSource {
  const core = createBrowserSensorSource({
    quaternion: false,
    absoluteQuaternion: false,
    orientationAbs: false,
    magnetometer: false,
    screenEvents: false,
  });
  let failed = false;
  return {
    kind: 'browser',
    engine,
    start(onSample: (s: LiveSample) => void, onFailure: (reason: SourceFailure) => void): Promise<void> {
      const check = createSensorCheck();
      const fail = (reason: SourceFailure): void => {
        if (failed) return;
        failed = true;
        core.stop();
        onFailure(reason);
      };
      return core.start({
        onMotion(s: MotionSample): void {
          if (failed) return;
          // an event without any value carries no data (desktop Chrome sends them): core/sensors.ts reports
          // no_events when nothing else arrives, and the engine never sees it
          if (s.acc === null && s.accg === null && s.rr === null) return;
          const verdict = check(s);
          if (verdict !== null && verdict !== 'ok') {
            fail(verdict);
            return;
          }
          onSample(s);
        },
        onOrientation(): void {
          // predict mode reads devicemotion only
        },
        onError(e: SensorError): void {
          fail(failureOfSensorError(e.code));
        },
      });
    },
    stop(): void {
      core.stop();
    },
  };
}
