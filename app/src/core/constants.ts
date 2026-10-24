import type {
  OrientationId,
  PhoneRotation,
  ScreenFacing,
  SegmentationParams,
  StudyConfig,
  Variant,
} from './types';

/** Keep in sync with package.json "version" (a unit test checks this). */
export const APP_VERSION = '0.2.1';
/** Data schema version (schema/records.schema.json). Minor bump = additive change. */
export const SCHEMA_VERSION = '1.2.0';
/**
 * Version of the protocol generator (presets, ordering rules). Bump when plans change: 2.0.0 added
 * the 'street' preset and removed the screen-away grips from every preset, so the same seed gives
 * a different plan than under 1.0.0.
 */
export const PROTOCOL_VERSION = '2.0.0';
/** v2 consent text: ~5 minutes (~30 movements + an optional bonus), no screen-away grips, no table step, names stored; v3: no bed/sofa advice, Hebrew fixes. */
export const CONSENT_VERSION_DEFAULT = '2026-10-v3';

export const GRAVITY = 9.80665;

/**
 * Default online segmentation parameters (protocol 2.0.0). Per-speed / per-variant overrides live
 * in protocol.ts (resolveParams). The online detector only drives cues and QA; the raw stream is
 * always kept so analyses can re-segment offline. The values are mirrored in
 * analysis/motionlab/conventions.py (a drift test parses this object).
 *
 * Changes from 1.0.0, for the hands-free street flow (from the Android pilot):
 *  - hold still at 20 deg/s and 0.5 m/s² for 400 ms (was 15, 0.35 and 600): the stricter values were
 *    broken by ordinary hand sway; in the pilot the time from the last large movement to 'still'
 *    drops from a median of 1.20 s to 0.47 s. Hold + foreperiod still give ≥ 0.8 s of rest before GO;
 *  - foreperiod 400 to 900 ms (was 500 to 1200);
 *  - end at 30 deg/s and 0.6 m/s² sustained 250 ms (was 20, 0.45 and 300); slow trials keep their
 *    10 / 0.25 / 450 override, which the looser end would cut mid-movement;
 *  - no onset within 3 s, or a movement over 6 s, fails the attempt (was 6 s and 8 s; slow and
 *    pause trials keep 12 s);
 *  - post-roll 500 ms is now a minimum: the controller keeps recording while the arm goes back out,
 *    until stillness after the return or return_cap_ms after the end was confirmed.
 * The simulated accuracy of both parameter sets is in app/scripts/out/segmentation_eval.md.
 */
export const DEFAULT_SEGMENTATION_PARAMS: SegmentationParams = {
  smoothing_ms: 40,
  gravity_tau_ms: 600,
  still_gyro_dps: 20,
  still_acc_ms2: 0.5,
  still_hold_ms: 400,
  foreperiod_min_ms: 400,
  foreperiod_max_ms: 900,
  onset_k_sd: 6,
  onset_gyro_min_dps: 30,
  onset_acc_min_ms2: 0.6,
  onset_sustain_ms: 60,
  end_gyro_dps: 30,
  end_acc_ms2: 0.6,
  end_sustain_ms: 250,
  min_movement_ms: 150,
  max_wait_ms: 3000,
  max_movement_ms: 6000,
  post_roll_ms: 500, // minimum after the end cue; the return of the arm is recorded after it
  return_cap_ms: 3000, // the recording stops at most this long after the end was confirmed
};

export interface OrientationInfo {
  rotation: PhoneRotation | null; // null for 'free'
  screen: ScreenFacing | null;
  /** orientation obtained by mirroring the body across the sagittal plane (left↔right hand) */
  mirror: OrientationId;
}

export const ORIENTATIONS: Record<OrientationId, OrientationInfo> = {
  upright_toward: { rotation: 'upright', screen: 'toward', mirror: 'upright_toward' },
  upright_away: { rotation: 'upright', screen: 'away', mirror: 'upright_away' },
  top_left_toward: { rotation: 'top_left', screen: 'toward', mirror: 'top_right_toward' },
  top_right_toward: { rotation: 'top_right', screen: 'toward', mirror: 'top_left_toward' },
  upside_down_toward: { rotation: 'upside_down', screen: 'toward', mirror: 'upside_down_toward' },
  upside_down_away: { rotation: 'upside_down', screen: 'away', mirror: 'upside_down_away' },
  top_left_away: { rotation: 'top_left', screen: 'away', mirror: 'top_right_away' },
  top_right_away: { rotation: 'top_right', screen: 'away', mirror: 'top_left_away' },
  flat_up: { rotation: 'flat', screen: 'up', mirror: 'flat_up' },
  free: { rotation: null, screen: null, mirror: 'free' },
};

export const ALL_ORIENTATIONS = Object.keys(ORIENTATIONS) as OrientationId[];

/**
 * The orientations the protocols collect (protocol 2.0.0), in ORIENTATIONS order: every id except the
 * screen-away grips, which participants found impractical; analyses synthesise screen-away data by
 * rotating the sensor frame. The *_away ids stay in the enum for older records and synthetic data.
 */
export const COLLECTED_ORIENTATIONS: OrientationId[] = ALL_ORIENTATIONS.filter((o) => ORIENTATIONS[o].screen !== 'away');

/** The four screen-toward grips of the brief (closed under mirroring), in the order they are introduced. */
export const CORE_ORIENTATIONS: OrientationId[] = ['upright_toward', 'top_left_toward', 'top_right_toward', 'upside_down_toward'];

export const ODD_VARIANTS: Exclude<Variant, 'standard'>[] = [
  'curved',
  'pause',
  'twist',
  'elbow_high',
  'bent_start',
  'lean_in',
  'chin',
  'forehead',
  'jerky',
  'tilted_touch',
];

export const DEFAULT_STUDY_CONFIG: StudyConfig = {
  study_code: 'nose-touch-v1',
  upload: { mode: 'none', endpoint: '', study_key: '' },
  // names are uploaded in this study (spec v2 decision 4; public/config.json says the same). The
  // fallback config uploads nothing (mode 'none'), and every upload is redacted again with the config
  // in force when it is sent, so a deployment that sets upload_nickname false never sends a name.
  upload_nickname: true,
  consent_version: CONSENT_VERSION_DEFAULT,
  contact: { name: '', email: '' },
  institution: { en: '', he: '' },
  region: { en: '', he: '' },
  default_language: 'en',
  default_preset: 'street',
  min_age: 18,
};

/** Rounding applied when serializing streams (keeps JSON ~2x smaller, far below sensor noise). */
export const ROUNDING = {
  time: 2, // 0.01 ms
  acc: 4, // 1e-4 m/s^2
  rr: 3, // 1e-3 deg/s
  angle: 3, // 1e-3 deg
  quat: 6,
} as const;
