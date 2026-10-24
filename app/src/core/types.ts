/**
 * Data contract of the collector app.
 *
 * Every persisted/uploaded type here mirrors `schema/records.schema.json` EXACTLY (same field names,
 * same nullability). `tests/schema.test.ts` validates generated records against that schema, so drift
 * fails the test suite. If you add a field: add it to the schema, bump SCHEMA_VERSION's minor digit.
 *
 * Units: time ms, acceleration m/s^2, rotation rate deg/s, angles deg.
 * Device frame (W3C): x → right edge of the screen, y → top edge, z → out of the screen (towards the
 * viewer of the screen). The frame is fixed to the device, NOT to the current screen rotation.
 */

// ───────────────────────────── enumerations ─────────────────────────────

export type Hand = 'left' | 'right';

/** How the phone is rotated as seen by the participant (from their own point of view). */
export type PhoneRotation = 'upright' | 'upside_down' | 'top_left' | 'top_right' | 'flat';
/** Which way the screen faces. 'toward' = towards the participant (main/rear camera faces away). */
export type ScreenFacing = 'toward' | 'away' | 'up';

export type OrientationId =
  | 'upright_toward'
  | 'upright_away'
  | 'top_left_toward'
  | 'top_right_toward'
  | 'upside_down_toward'
  | 'upside_down_away'
  | 'top_left_away'
  | 'top_right_away'
  | 'flat_up'
  | 'free';

export type Speed = 'natural' | 'fast' | 'slow';
/** natural = participant's own reading of "arm straight in front"; shoulder = hand in line with the same shoulder; center = phone in front of the face. */
export type StartPosition = 'natural' | 'shoulder' | 'center';
export type Variant =
  | 'standard'
  | 'curved'
  | 'pause'
  | 'twist'
  | 'elbow_high'
  | 'bent_start'
  | 'lean_in'
  | 'chin'
  | 'forehead'
  | 'jerky'
  /** schema 1.2: touch the nose with the top edge of the phone, tilting it as it arrives */
  | 'tilted_touch';
export type Posture = 'standing' | 'sitting';
/** 'street' (schema 1.2, protocol 2.0.0): the ~5 minute participant visit in priority tiers (the default). */
export type ProtocolPreset = 'quick' | 'standard' | 'full' | 'custom' | 'street';
/**
 * Kind of a planned block. 'choice' (schema 1.2): a single trial whose hand the participant picks and
 * reports afterwards; the block's planned hand is only a placeholder.
 */
export type BlockKind = 'main' | 'odd' | 'free' | 'choice';
/** schema 1.2: who decided the hand of a trial: the protocol ('instructed') or the participant ('reported'). */
export type HandSource = 'instructed' | 'reported';
/** schema 1.2: 'auto' = complete trials are accepted without a review screen; 'tap' = every trial is reviewed. */
export type ReviewMode = 'auto' | 'tap';
/** schema 1.2: 'ended_early' = the participant chose Finish before every non-optional block was done. */
export type SessionStatus = 'in_progress' | 'completed' | 'abandoned' | 'ended_early';
/** schema 1.2: guided_v1 = the calibration screen; implicit_v1 = derived from the first still upright trial. */
export type CalibrationMethod = 'guided_v1' | 'implicit_v1';
export type Language = 'en' | 'he';
export type CueKind = 'ready' | 'go' | 'end' | 'error' | 'block_start';
export type Sex = 'female' | 'male' | 'other' | 'prefer_not_to_say';
export type HandednessSelf = 'right' | 'left' | 'ambidextrous';
export type EhiAnswer = 'always_right' | 'usually_right' | 'both' | 'usually_left' | 'always_left';
export type EhiClass = 'right' | 'mixed' | 'left';
export type PhoneHand = 'right' | 'left' | 'both';
export type Platform = 'ios' | 'android' | 'other';
export type TrialStatus = 'complete' | 'no_movement' | 'too_long' | 'aborted' | 'interrupted';
export type ParticipantFlag = 'wrong_hand' | 'wrong_orientation' | 'other_problem';
export type PermissionState = 'granted' | 'denied' | 'not_required' | 'unknown';

export type Vec3 = [number, number, number];
/**
 * W3C {alpha, beta, gamma} triple. CAREFUL — the two W3C interfaces use DIFFERENT axis conventions:
 *  - DeviceMotionEvent.rotationRate (deg/s): alpha = rate about device X, beta = about Y, gamma = about Z
 *    (W3C CRD 12 Feb 2025, "The alpha getter steps are to return the value of this's x axis rotation
 *    rate"; changed in 2019 to match Chrome/Firefox/Safari. MDN's "alpha = Z" text is outdated).
 *  - DeviceOrientationEvent angles (deg): alpha about Z, then beta about X', then gamma about Y''.
 */
export interface Euler {
  alpha: number;
  beta: number;
  gamma: number;
}

// ───────────────────────────── protocol ─────────────────────────────

export interface TrialCondition {
  hand: Hand;
  orientation: OrientationId;
  speed: Speed;
  start_position: StartPosition;
  variant: Variant;
  posture: Posture;
  /** repetition index of this exact condition within its block (0-based) */
  rep: number;
  /**
   * optional (schema 1.2; protocol 2.0.0 plans always set it): 'reported' for the trial of a 'choice'
   * block, whose `hand` is replaced by the participant's answer after the trial; 'instructed' otherwise.
   * Absent in older records = instructed.
   */
  hand_source?: HandSource;
}

export interface PlannedTrial {
  trial_index: number; // global, 0-based, unique within the session plan
  block_index: number;
  trial_in_block: number;
  condition: TrialCondition;
}

export interface PlannedBlock {
  block_index: number;
  kind: BlockKind;
  hand: Hand;
  orientation: OrientationId;
  trials: PlannedTrial[];
  /**
   * optional (schema 1.2; set on every block of a 'street' plan): priority tier, lower tiers run first.
   * street: 0 choice, 1 basics, 2 sideways, 3 variety, 4 bonus. Absent for the other presets.
   */
  tier?: number;
  /**
   * optional (schema 1.2; set on every block of a 'street' plan): true = bonus block, run only when the
   * participant opts in at the checkpoint; a session is complete without it. Absent = false.
   */
  optional?: boolean;
}

export interface CustomProtocolSpec {
  orientations: OrientationId[];
  speeds: Speed[];
  start_positions: StartPosition[];
  reps: number;
  odd_variants: Variant[];
  include_free: boolean;
}

export interface ProtocolOptions {
  preset: ProtocolPreset;
  seed: number; // uint32
  posture: Posture;
  /** if omitted, derived from the seed (balances first hand across participants) */
  firstHand?: Hand;
  custom?: CustomProtocolSpec | null;
}

// ───────────────────────────── segmentation ─────────────────────────────

export interface SegmentationParams {
  smoothing_ms: number; // EMA time constant applied to |ω| and |a_lin|
  gravity_tau_ms: number; // EMA time constant of the gravity estimate (high-pass fallback when 'acceleration' is null)
  still_gyro_dps: number; // hold-still: smoothed |ω| must stay below
  still_acc_ms2: number; // hold-still: smoothed |a_lin| must stay below
  still_hold_ms: number; // required continuous stillness before the GO cue is scheduled
  foreperiod_min_ms: number; // random delay between stillness and GO (anti-anticipation)
  foreperiod_max_ms: number;
  onset_k_sd: number; // adaptive threshold = max(floor, baseline_mean + k * baseline_sd)
  onset_gyro_min_dps: number; // floor of the gyro onset threshold
  onset_acc_min_ms2: number; // floor of the accel onset threshold
  onset_sustain_ms: number; // threshold must be exceeded continuously this long
  end_gyro_dps: number; // end: smoothed |ω| below
  end_acc_ms2: number; // end: smoothed |a_lin| below
  end_sustain_ms: number; // ... continuously for this long
  min_movement_ms: number; // an end cannot be declared before onset + this
  max_wait_ms: number; // no onset within this after GO → 'no_movement'
  max_movement_ms: number; // movement still going after this → 'too_long'
  post_roll_ms: number; // keep recording at least this long after the end is confirmed
  /**
   * optional (schema 1.2; DEFAULT_SEGMENTATION_PARAMS always has it): after the end is confirmed, keep
   * recording through the return of the arm (return movement, then stillness) for at most this long.
   * Read by the trial controller only; the online segmenter ignores it.
   */
  return_cap_ms?: number;
}

export interface Baseline {
  n: number;
  gyro_mean: number;
  gyro_sd: number;
  acc_mean: number;
  acc_sd: number;
  gravity: Vec3 | null; // mean accelerationIncludingGravity during the still window (raw device frame)
}

export interface SegmentationResult {
  method: 'online_v1';
  params: SegmentationParams;
  acc_source: 'linear' | 'highpass' | 'none';
  onset_index: number | null; // index into streams.motion arrays
  end_index: number | null;
  duration_ms: number | null;
  peak_gyro_dps: number | null;
  peak_acc_ms2: number | null;
  baseline: Baseline | null;
}

// ───────────────────────────── participant / device ─────────────────────────────

export interface EhiShortForm {
  writing: EhiAnswer;
  throwing: EhiAnswer;
  toothbrush: EhiAnswer;
  spoon: EhiAnswer;
}

export interface ParticipantProfile {
  participant_id: string; // "P-XXXXXX", alphabet without 0/O/1/I
  nickname: string | null; // kept on the device; uploaded only if StudyConfig.upload_nickname
  age_years: number | null;
  sex: Sex;
  handedness_self: HandednessSelf;
  ehi_sf: EhiShortForm | null;
  ehi_lq: number | null; // Edinburgh short-form laterality quotient, -100..100
  ehi_class: EhiClass | null;
  phone_hand: PhoneHand; // which hand usually holds the phone
  height_cm: number | null;
  arm_issue: boolean | null; // any condition affecting arm/hand movement
  phone_model: string | null;
  phone_case: boolean | null;
  created_at: string;
  updated_at: string;
}

export interface DeviceInfo {
  user_agent: string;
  platform: Platform;
  browser: string;
  os_version: string | null;
  model_hint: string | null; // e.g. from UA Client Hints on Android; iOS does not expose models
  standalone: boolean; // launched from the home screen
  screen_w: number;
  screen_h: number;
  dpr: number;
  language: string;
  hardware_concurrency: number | null;
  device_memory_gb: number | null;
  time_origin_ms: number; // performance.timeOrigin
  /** optional (schema 1.1). webkit = every iOS browser: acceleration values are sign-inverted vs W3C. */
  engine?: 'webkit' | 'blink' | 'gecko' | 'unknown';
  /** optional (schema 1.1): detected in-app browser / webview, e.g. 'instagram', 'android_webview' */
  in_app_browser?: string | null;
}

export interface MeasuredRates {
  duration_ms: number;
  motion_hz: number | null;
  orientation_hz: number | null;
  quaternion_hz: number | null;
  motion_dt_ms_median: number | null;
  motion_dt_ms_p95: number | null;
  acc_available: boolean;
  accg_available: boolean;
  rotation_available: boolean;
  interval_raw: number | null;
}

export interface SensorCapabilities {
  secure_context: boolean;
  device_motion: boolean;
  device_orientation: boolean;
  permission_api: boolean; // iOS-style DeviceMotionEvent.requestPermission exists
  permission_state: PermissionState;
  generic_relative_orientation: boolean;
  wake_lock: boolean;
  vibrate: boolean;
  share_files: boolean;
  measured: MeasuredRates | null;
}

export interface PoseStats {
  n: number;
  duration_ms: number;
  accg_mean: Vec3;
  accg_sd: Vec3;
  acc_mean: Vec3 | null;
  rr_mean: Euler | null;
  rr_sd: Euler | null;
  still: boolean;
}

export interface CalibrationResult {
  performed_at: string;
  flat: PoseStats | null; // phone flat on a table, screen up
  upright: PoseStats | null; // phone upright in front of the participant, screen towards them
  /** +1: face-up reads accg_z ≈ +9.81 (W3C convention). -1: inverted. null: unknown. */
  gravity_sign: 1 | -1 | null;
  gravity_magnitude: number | null;
  upright_check: 'ok' | 'inverted' | 'unclear' | null;
  gyro_bias_dps: Euler | null;
  notes: string[];
  /** optional (schema 1.1): empirical rotationRate axis/sign/unit check from a tilt gesture */
  axis_check?: AxisCheck | null;
  /** optional (schema 1.2): how the calibration was obtained. Absent = 'guided_v1'. */
  method?: CalibrationMethod;
}

/**
 * Tilt-gesture check of the rotationRate convention. The device rotation axis is observed from the
 * gravity direction (accelerometer; independent of its sign convention because both vectors flip
 * together) and compared with the integrated rotationRate under each candidate field mapping.
 */
export interface AxisCheck {
  n: number;
  duration_ms: number;
  tilt_deg: number | null;
  device_axis: Vec3 | null;
  rr_integral_deg: Euler | null;
  /** w3c: (alpha,beta,gamma) = (x,y,z). legacy: (beta,gamma,alpha) = (x,y,z), i.e. alpha about Z. */
  verdict: 'w3c' | 'w3c_inverted' | 'legacy' | 'legacy_inverted' | 'unclear';
  units: 'deg_per_s' | 'rad_per_s' | null;
  similarity: number | null;
  magnitude_ratio: number | null;
}

// ───────────────────────────── records ─────────────────────────────

export interface SessionRecord {
  record_type: 'session';
  schema_version: string;
  app_version: string;
  session_id: string;
  participant_id: string;
  study_code: string;
  started_at: string;
  updated_at: string;
  ended_at: string | null;
  status: SessionStatus;
  participant: ParticipantProfile; // snapshot, already redacted per StudyConfig
  device: DeviceInfo;
  capabilities: SensorCapabilities;
  calibration: CalibrationResult | null;
  protocol: {
    preset: ProtocolPreset;
    protocol_version: string;
    seed: number;
    posture: Posture;
    first_hand: Hand;
    n_planned_trials: number;
    estimated_minutes: number;
    custom: CustomProtocolSpec | null;
    plan: PlannedBlock[];
  };
  settings: {
    language: Language;
    audio: boolean;
    speech: boolean;
    segmentation_params: SegmentationParams; // base params (per-trial overrides are stored in each trial)
    /** optional (schema 1.2; the v2 app always writes it): how trials were accepted. Absent = 'tap'. */
    review_mode?: ReviewMode;
  };
  progress: {
    next_trial_index: number; // index into the flattened plan of the next trial to run
    n_attempts: number;
    n_accepted: number;
  };
  consent: { version: string; accepted_at: string };
  timezone_offset_min: number;
  simulated: boolean;
  notes: string | null;
}

export interface MotionStream {
  t: number[];
  ts: (number | null)[];
  acc_x: (number | null)[];
  acc_y: (number | null)[];
  acc_z: (number | null)[];
  accg_x: (number | null)[];
  accg_y: (number | null)[];
  accg_z: (number | null)[];
  rr_alpha: (number | null)[];
  rr_beta: (number | null)[];
  rr_gamma: (number | null)[];
  interval_raw: number | null;
  ts_first_raw: number | null;
}

export interface OrientationStream {
  t: number[];
  ts: (number | null)[];
  alpha: (number | null)[];
  beta: (number | null)[];
  gamma: (number | null)[];
  compass: (number | null)[];
  absolute: boolean | null;
  ts_first_raw: number | null;
  /** optional (schema 1.2): iOS webkitCompassAccuracy (deg; negative = invalid), one entry per sample */
  compass_accuracy?: (number | null)[];
}

export interface QuaternionStream {
  t: number[];
  x: number[];
  y: number[];
  z: number[];
  w: number[];
  sensor: 'RelativeOrientationSensor' | 'AbsoluteOrientationSensor';
  frequency_hz: number;
}

/**
 * Contact details of the touch whose clientX/clientY are stored (schema 1.2, all optional in records):
 * Touch.radiusX / radiusY (CSS px), force (0..1; 0 where not measured), rotationAngle (deg), and the
 * position normalised by the viewport (x_norm = clientX / innerWidth, y_norm = clientY / innerHeight).
 */
export interface TouchDetails {
  radius_x: number | null;
  radius_y: number | null;
  force: number | null;
  rotation_angle: number | null;
  x_norm: number | null;
  y_norm: number | null;
}

export interface TouchRecord extends Partial<TouchDetails> {
  t: number;
  type: 'start' | 'end' | 'cancel';
  n: number;
  x: number | null;
  y: number | null;
}

/** schema 1.2: a screen orientation change during the trial (t relative to t0, like every stream). */
export interface ScreenEventRecord {
  t: number;
  type: string | null; // screen.orientation.type after the change
  angle: number | null; // screen.orientation.angle after the change (deg)
}

/** schema 1.2: Generic Sensor API Magnetometer readings, µT in the device frame. */
export interface MagnetometerStream {
  t: number[];
  x: (number | null)[];
  y: (number | null)[];
  z: (number | null)[];
  frequency_hz: number;
}

export interface CueRecord {
  t: number;
  kind: CueKind;
}

export interface TrialTiming {
  t0_perf_ms: number;
  t0_epoch_ms: number;
  still_ms: number | null;
  go_ms: number | null;
  onset_ms: number | null;
  end_ms: number | null;
  end_detected_ms: number | null;
  stop_ms: number;
  first_touch_ms: number | null;
  foreperiod_ms: number | null;
  false_starts: number;
  /** optional (schema 1.2; the v2 app always writes it): start of the return movement after the end; null = not observed */
  return_onset_ms?: number | null;
  /** optional (schema 1.2; the v2 app always writes it): still again after the return; null = not observed */
  return_end_ms?: number | null;
}

export interface TrialDetected {
  rotation_start: PhoneRotation | 'unknown';
  rotation_end: PhoneRotation | 'unknown';
  tilt_change_deg: number | null;
  orientation_match: boolean | null;
}

export interface TrialQuality {
  n_motion: number;
  n_orientation: number;
  n_quaternion: number;
  motion_hz: number | null;
  motion_dt_ms_median: number | null;
  motion_dt_ms_max: number | null;
  gaps_over_50ms: number;
  acc_available: boolean;
  rotation_available: boolean;
  hidden_during: boolean;
  warnings: string[];
  /** optional (schema 1.2): samples in streams.quaternion_abs */
  n_quaternion_abs?: number;
  /** optional (schema 1.2): samples in streams.orientation_abs */
  n_orientation_abs?: number;
}

export interface TrialOutcome {
  status: TrialStatus;
  accepted: boolean;
  participant_flag: ParticipantFlag | null;
  note: string | null;
}

export interface TrialRecord {
  record_type: 'trial';
  schema_version: string;
  app_version: string;
  trial_id: string;
  session_id: string;
  participant_id: string;
  study_code: string;
  created_at: string;
  block_index: number;
  trial_index: number;
  trial_in_block: number;
  attempt: number; // 1 = first attempt of this planned trial; redo → 2, 3, ...
  simulated: boolean;
  condition: TrialCondition;
  outcome: TrialOutcome;
  timing: TrialTiming;
  segmentation: SegmentationResult;
  detected: TrialDetected;
  quality: TrialQuality;
  screen: { type_start: string | null; angle_start: number | null; type_end: string | null; angle_end: number | null };
  streams: {
    motion: MotionStream;
    orientation: OrientationStream | null;
    quaternion: QuaternionStream | null;
    touches: TouchRecord[];
    cues: CueRecord[];
    // ── optional (schema 1.2; best effort, the v2 app writes them where the browser offers them) ──
    /** AbsoluteOrientationSensor (magnetometer-fused, Earth-referenced); null when no sample */
    quaternion_abs?: QuaternionStream | null;
    /** 'deviceorientationabsolute' events (Chrome Android); null when no event */
    orientation_abs?: OrientationStream | null;
    /** screen orientation changes during the trial */
    screen_events?: ScreenEventRecord[];
    /** Magnetometer readings where exposed (usually behind a browser flag); null otherwise */
    magnetometer?: MagnetometerStream | null;
  };
}

export interface UploadEnvelope {
  kind: 'session' | 'trial';
  study_key: string | null;
  sent_at: string;
  record: SessionRecord | TrialRecord;
}

export interface ExportBundle {
  format: 'nose-touch-export';
  schema_version: string;
  app_version: string;
  exported_at: string;
  participants: ParticipantProfile[];
  sessions: SessionRecord[];
  trials: TrialRecord[];
}

// ───────────────────────────── study config (public/config.json) ─────────────────────────────

export type UploadMode = 'apps_script' | 'json_post' | 'none';

export interface UploadConfig {
  mode: UploadMode;
  endpoint: string; // Apps Script /exec URL or any HTTPS endpoint accepting UploadEnvelope JSON
  study_key: string; // shared token checked by the backend (spam filter, not real security)
}

/** A text the researcher sets once per UI language (e.g. an institution's name). */
export type LocalizedText = Record<Language, string>;

export interface StudyConfig {
  study_code: string; // [A-Za-z0-9_-]
  upload: UploadConfig;
  upload_nickname: boolean;
  consent_version: string;
  contact: { name: string; email: string };
  /** Named in the consent/about texts, e.g. {en: 'Bar-Ilan University', he: 'אוניברסיטת בר-אילן'}.
   *  Empty → neutral wording ("Researchers are studying…"). Set it only once the study is approved. */
  institution: LocalizedText;
  /** Where participants must be located, e.g. {en: 'Israel', he: 'ישראל'}; empty → no location clause. */
  region: LocalizedText;
  default_language: Language;
  default_preset: ProtocolPreset;
  min_age: number;
}

// ───────────────────────────── runtime (not persisted as-is) ─────────────────────────────

/** One devicemotion reading, as delivered by a SensorSource (times in the source clock, NOT relative). */
export interface MotionSample {
  t: number; // source clock (performance.now() for real sensors), ms
  ts: number | null; // raw event.timeStamp
  acc: Vec3 | null; // acceleration (gravity excluded), null if the platform does not provide it
  accg: Vec3 | null; // accelerationIncludingGravity
  rr: Euler | null; // rotationRate, deg/s: alpha = about device X, beta = about Y, gamma = about Z
  interval: number | null; // raw event.interval
}

export interface OrientationSample {
  t: number;
  ts: number | null;
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
  absolute: boolean | null;
  compass: number | null;
  /** schema 1.2: iOS webkitCompassAccuracy (deg; negative = invalid); undefined/null where absent */
  compass_accuracy?: number | null;
}

export interface QuaternionSample {
  t: number;
  q: [number, number, number, number]; // x, y, z, w
  sensor: 'RelativeOrientationSensor' | 'AbsoluteOrientationSensor';
}

/** schema 1.2: one Magnetometer reading (µT, device frame), source clock. */
export interface MagnetometerSample {
  t: number;
  x: number | null;
  y: number | null;
  z: number | null;
}

/** schema 1.2: screen orientation after a 'change' event, source clock. */
export interface ScreenSample {
  t: number;
  type: string | null;
  angle: number | null;
}

export interface SensorError {
  code: 'not_supported' | 'permission_denied' | 'insecure_context' | 'no_events' | 'sensor_error';
  message: string;
}

export interface SensorListener {
  onMotion(s: MotionSample): void;
  onOrientation(s: OrientationSample): void;
  /** RelativeOrientationSensor readings (sensor 'RelativeOrientationSensor') */
  onQuaternion?(s: QuaternionSample): void;
  /** schema 1.2: 'deviceorientationabsolute' events (absolute = true) → streams.orientation_abs */
  onOrientationAbs?(s: OrientationSample): void;
  /** schema 1.2: AbsoluteOrientationSensor readings (sensor 'AbsoluteOrientationSensor') → streams.quaternion_abs */
  onQuaternionAbs?(s: QuaternionSample): void;
  /** schema 1.2: Magnetometer readings → streams.magnetometer */
  onMagnetometer?(s: MagnetometerSample): void;
  /** schema 1.2: screen orientation 'change' events → streams.screen_events */
  onScreen?(s: ScreenSample): void;
  onError?(e: SensorError): void;
}

/** Time base of a sensor source. Real sensors: performance.now(); simulator: virtual time. */
export interface SourceClock {
  now(): number;
  /** Unix epoch ms for a time expressed in this clock. */
  epochAt(t: number): number;
}

export interface SensorSource {
  readonly kind: 'browser' | 'simulated';
  readonly clock: SourceClock;
  /** Starts delivering samples to `listener`. A simulated source in offline mode may deliver ALL samples synchronously inside start() until stop() is called from a listener callback. */
  start(listener: SensorListener): Promise<void>;
  stop(): void;
  /** Lets a simulated participant react to app cues (GO → start moving). No-op for real sensors. */
  notifyCue?(cue: CueKind, t: number): void;
}

/** Plays audio/haptic/speech cues. Implementations must never throw. */
export interface CueSink {
  play(kind: CueKind): void;
  /** true when an utterance was actually queued (speech on, a voice engine, not silenced) */
  say?(text: string, lang: Language): boolean | void;
}

/**
 * 'returning' (v2): after the end is confirmed and post_roll_ms has passed, the recording continues
 * while the participant brings the arm back out (until stillness after a return movement, or
 * return_cap_ms after end_detected).
 */
export type TrialPhase = 'waiting_still' | 'foreperiod' | 'armed' | 'moving' | 'post_roll' | 'returning' | 'done';

export type TrialMessage =
  | 'hold_still'
  | 'too_early'
  | 'go'
  | 'moving'
  | 'hold_at_nose'
  | 'done'
  | 'no_movement'
  | 'too_long'
  | 'sensor_stalled'
  /** v2: stretch the arm out again (after the end cue, or before GO while awaiting the return) */
  | 'return_arm'
  /** v2 grip gate: the phone is held in a different known grip than instructed; GO is withheld */
  | 'wrong_grip'
  /** v2 reading time (TrialControllerOptions.readMs): GO is withheld while the new instruction is read */
  | 'read';

export interface TrialLiveState {
  phase: TrialPhase;
  t: number; // ms since t0
  gyro_dps: number; // smoothed |ω|
  acc_ms2: number; // smoothed |a_lin|
  still_progress: number; // 0..1 of still_hold_ms
  message: TrialMessage | null;
  /**
   * v2 grip gate: how the phone was held at the last gate check (gravity over the last 300 ms, under the
   * detection calibration's sign); null while the gate has not judged a grip (no gate, not still yet).
   */
  grip_seen?: PhoneRotation | 'unknown' | null;
}

export interface TrialIds {
  trial_id: string;
  session_id: string;
  participant_id: string;
  study_code: string;
  block_index: number;
  trial_index: number;
  trial_in_block: number;
  attempt: number;
}

export interface TrialControllerOptions {
  source: SensorSource;
  cues: CueSink;
  ids: TrialIds;
  condition: TrialCondition;
  /** already resolved with per-speed/variant overrides (protocol.resolveParams) */
  params: SegmentationParams;
  calibration: CalibrationResult | null;
  rng: () => number; // [0,1)
  appVersion: string;
  simulated: boolean;
  /** called on every motion sample; UI should throttle its own re-renders */
  onLive?: (s: TrialLiveState) => void;
  /** returns screen.orientation {type, angle} or nulls; injected so core stays DOM-free */
  getScreenOrientation?: () => { type: string | null; angle: number | null };
  /** true if document is currently hidden; injected so core stays DOM-free */
  isHidden?: () => boolean;
  /**
   * v2 chained trials: stillness only counts (GO can be scheduled) after a return movement has been
   * seen in this trial, because the previous trial's recording ended before the arm was back out.
   * Default false.
   */
  awaitReturn?: boolean;
  /** play the 'ready' cue at the start of the trial. Default true; chained trials pass false. */
  readyCue?: boolean;
  /**
   * v2 grip gate: at 'still', withhold GO (message 'wrong_grip') while the gravity direction confidently
   * matches a different known grip than the instructed one. Default false; never applies to 'free'.
   */
  gripGate?: boolean;
  /**
   * v2 reading time (ms after t0): GO is never scheduled before it (message 'read' while nothing more
   * urgent is shown), so a new instruction on the screen — or the words being spoken — can be taken in
   * before the beep. still_ms is then ≥ readMs. Default 0 (none).
   */
  readMs?: number;
}

/** Local outbox entry for the upload queue (IndexedDB). */
export interface OutboxItem {
  id: string; // `${kind}:${record_id}`
  kind: 'session' | 'trial';
  record_id: string;
  session_id: string;
  status: 'pending' | 'sent' | 'failed';
  attempts: number;
  last_error: string | null;
  last_attempt_at: string | null;
  sent_at: string | null;
  confirmed: boolean; // response was readable and said ok (false for opaque no-cors fallbacks)
  /**
   * The server refused this record ({ok: false} without retry: true, e.g. a wrong study key): status
   * is 'failed', and only a forced flush ("Upload now") tries it again. Absent otherwise; a
   * successful upload or a re-enqueue (new content) clears it.
   */
  permanent?: boolean;
}
