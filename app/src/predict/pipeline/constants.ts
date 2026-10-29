/**
 * Constants of the standardisation pipeline. Python ground truth (paths relative to analysis/):
 *   stage2/config.py:110-116      FS_HZ 60, LOWPASS_HZ 15, LOWPASS_ORDER 4, MAX_GAP_MS 100, STILL_GYRO_DPS 20,
 *                                 REST_MS 300, N_NORM 64
 *   motionlab/preprocess.py       gyro_gravity (min_rest_n 5, |g_pre| window 8.6..11.0, calibration magnitude
 *                                 window 8.0..11.5), repeated_sample_mask (chrome_min_dps 20)
 *   motionlab/conventions.py:66-68  GRAVITY, DEG2RAD, RAD2DEG
 *   stage2/standardize.py:197     fewer than 5 kept rows -> ValueError
 *
 * The Butterworth coefficients are scipy's, never designed here: signal.butter(4, 15, 'low', fs=60,
 * output='sos') and signal.sosfilt_zi of it, exactly as analysis/stage3 exported them into
 * models/m4_production.json (detector.lowpassSos / lowpassZi / lowpassPadlen). constants.test.ts checks that
 * these literals equal the exported values, so the pipeline needs no import of the model bundle.
 */
import type { CanonicalChannel } from '../types';

/** stream sample rate (config.FS_HZ) */
export const FS_HZ = 60;
/** uniform_grid's step: 1000.0 / 60 = 16.666666666666668 (preprocess.py:60) */
export const STEP_MS = 1000.0 / FS_HZ;
/** grid points inside a gap between two finite samples longer than this stay NaN (config.MAX_GAP_MS) */
export const MAX_GAP_MS = 100.0;

/** sos rows [b0, b1, b2, a0, a1, a2]; the two a1 values near 1e-16 are scipy's, not zero (PORT_SPEC 1.7) */
export const LOWPASS_SOS: readonly (readonly number[])[] = [
  [0.09398085143379445, 0.1879617028675889, 0.09398085143379445, 1, -1.387778780781446e-16, 0.03956612989658006],
  [1, 2, 1, 1, -1.1102230246251565e-16, 0.4464626921716894],
];
/** sosfilt_zi(LOWPASS_SOS), rows [z0, z1] */
export const LOWPASS_ZI: readonly (readonly number[])[] = [
  [0.26763482160912794, 0.07967311874153896],
  [0.6383843269570777, -0.0848470191287671],
];
/** sosfiltfilt's default pad length 3 * (2 * n_sections + 1 - min(#b2 == 0, #a2 == 0)) = 15 (preprocess.py:136) */
export const LOWPASS_PADLEN = 15;

/** gravity_v1 rest windows: |w| below this is still (config.STILL_GYRO_DPS) */
export const STILL_GYRO_DPS = 20.0;
/** gravity_v1 rest window length before the onset and after the end (config.REST_MS); widened to 2x when short */
export const REST_MS = 300.0;
/** a rest window needs at least this many samples (gyro_gravity min_rest_n) */
export const MIN_REST_N = 5;
/** |g_pre| is used as the gravity magnitude only inside [MAG_MIN, MAG_MAX] (preprocess.py:540) */
export const MAG_MIN = 8.6;
export const MAG_MAX = 11.0;
/** a calibration magnitude is accepted only inside [CAL_MAG_MIN, CAL_MAG_MAX] (preprocess.py:538; never set in production) */
export const CAL_MAG_MIN = 8.0;
export const CAL_MAG_MAX = 11.5;
/** standard gravity (conventions.GRAVITY) */
export const GRAVITY = 9.80665;
/** time_normalised length (config.N_NORM) */
export const N_NORM = 64;
/** stream_from_record raises below this many kept rows (standardize.py:197) */
export const MIN_ROWS = 5;
/** non-WebKit engines drop an isolated repeat only while |rr| exceeds this (preprocess.py:576) */
export const REPEAT_MIN_DPS = 20.0;

/** conventions.DEG2RAD = math.pi / 180.0 */
export const DEG2RAD = Math.PI / 180.0;
/** conventions.RAD2DEG = 180.0 / math.pi; also numpy's np.degrees factor (npy_rad2deg: x * (180.0 / NPY_PI)) */
export const RAD2DEG = 180.0 / Math.PI;

/** standardize.CHANNELS (standardize.py:61-64): the row order of every (17, T) array */
export const CHANNELS: readonly CanonicalChannel[] = [
  'acc_x',
  'acc_y',
  'acc_z',
  'gyro_x',
  'gyro_y',
  'gyro_z',
  'up_x',
  'up_y',
  'up_z',
  'gyro_mag',
  'acc_mag',
  'acc_vert',
  'acc_horiz',
  'yaw_rate',
  'gyro_horiz',
  'yaw_cum',
  'tilt_from_start',
];
export const N_CHANNELS = 17;

/** row index of a channel (standardize.channel_index) */
export function channelIndex(name: CanonicalChannel): number {
  const i = CHANNELS.indexOf(name);
  if (i < 0) throw new RangeError(`unknown channel ${name}`);
  return i;
}

/** Index of yaw_cum (re-zeroed by time_normalised). */
export const YAW_CUM = 15;
