import type {
  OrientationId,
  PhoneRotation,
  ScreenFacing,
  SegmentationParams,
  StudyConfig,
  Variant,
} from './types';

/** Keep in sync with package.json "version" (a unit test checks this). */
export const APP_VERSION = '0.1.0';
/** Data schema version (schema/records.schema.json). Minor bump = additive change. */
export const SCHEMA_VERSION = '1.1.0';
/** Version of the protocol generator (presets, ordering rules). Bump when plans change. */
export const PROTOCOL_VERSION = '1.0.0';
export const CONSENT_VERSION_DEFAULT = '2026-10-v1';

export const GRAVITY = 9.80665;

/**
 * Default online segmentation parameters. Per-speed / per-variant overrides live in
 * protocol.ts (resolveParams). Values are deliberately conservative: the online detector only
 * drives cues and QA; the raw stream is always kept so analyses can re-segment offline.
 */
export const DEFAULT_SEGMENTATION_PARAMS: SegmentationParams = {
  smoothing_ms: 40,
  gravity_tau_ms: 600,
  // 15 deg/s / 0.35 m/s² are kept until pilot data on real phones exist: 10 / 0.25 improved the
  // 95th-percentile onset error in simulation (428 → 331 ms) but the simulator's tremor is calmer than
  // real hands, so the cost (participants failing to hold still) is unknown (app/scripts/out/
  // segmentation_eval.md). The hold + foreperiod below guarantee 1.1–1.8 s of stillness before GO.
  still_gyro_dps: 15,
  still_acc_ms2: 0.35,
  still_hold_ms: 600,
  foreperiod_min_ms: 500,
  foreperiod_max_ms: 1200,
  onset_k_sd: 6,
  onset_gyro_min_dps: 30,
  onset_acc_min_ms2: 0.6,
  onset_sustain_ms: 60,
  end_gyro_dps: 20,
  end_acc_ms2: 0.45,
  end_sustain_ms: 300,
  min_movement_ms: 150,
  max_wait_ms: 6000,
  max_movement_ms: 8000,
  post_roll_ms: 1000, // ≥ 1 s of post-movement rest for offline re-segmentation and gravity anchors
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

/** The five orientations named in the brief, in the order they are introduced to participants. */
export const CORE_ORIENTATIONS: OrientationId[] = [
  'upright_toward',
  'upright_away',
  'top_left_toward',
  'top_right_toward',
  'upside_down_toward',
];

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
];

export const DEFAULT_STUDY_CONFIG: StudyConfig = {
  study_code: 'nose-touch-v1',
  upload: { mode: 'none', endpoint: '', study_key: '' },
  upload_nickname: false,
  consent_version: CONSENT_VERSION_DEFAULT,
  contact: { name: '', email: '' },
  institution: { en: '', he: '' },
  region: { en: '', he: '' },
  default_language: 'en',
  default_preset: 'standard',
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
