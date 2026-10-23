/**
 * Hand-built, schema-valid records for the data-layer tests (storage, uploader, exporter, backend).
 * Deliberately independent of the recorder/protocol modules so these tests never depend on them.
 */
import { APP_VERSION, DEFAULT_SEGMENTATION_PARAMS, SCHEMA_VERSION } from '../../src/core/constants';
import type {
  ExportBundle,
  ParticipantProfile,
  SessionRecord,
  TrialCondition,
  TrialRecord,
  UploadConfig,
  UploadEnvelope,
} from '../../src/core/types';

export const T0 = '2026-10-23T10:00:00.000Z';

export function makeProfile(over: Partial<ParticipantProfile> = {}): ParticipantProfile {
  return {
    participant_id: 'P-ABCDEF',
    nickname: 'Dana',
    age_years: 31,
    sex: 'female',
    handedness_self: 'right',
    ehi_sf: { writing: 'always_right', throwing: 'always_right', toothbrush: 'usually_right', spoon: 'always_right' },
    ehi_lq: 87.5,
    ehi_class: 'right',
    phone_hand: 'right',
    height_cm: 168,
    arm_issue: false,
    phone_model: null,
    phone_case: true,
    created_at: T0,
    updated_at: T0,
    ...over,
  };
}

export function makeCondition(over: Partial<TrialCondition> = {}): TrialCondition {
  return {
    hand: 'right',
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
    ...over,
  };
}

export function makeSession(over: Partial<SessionRecord> = {}): SessionRecord {
  const participant = over.participant ?? makeProfile(over.participant_id ? { participant_id: over.participant_id } : {});
  return {
    record_type: 'session',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    session_id: 'sess-0001',
    participant_id: participant.participant_id,
    study_code: 'nose-touch-v1',
    started_at: T0,
    updated_at: T0,
    ended_at: null,
    status: 'in_progress',
    participant,
    device: {
      user_agent: 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 Chrome/141.0 Mobile Safari/537.36',
      platform: 'android',
      browser: 'chrome',
      os_version: '15',
      model_hint: 'Pixel 7',
      standalone: false,
      screen_w: 412,
      screen_h: 915,
      dpr: 2.625,
      language: 'en-US',
      hardware_concurrency: 8,
      device_memory_gb: 8,
      time_origin_ms: 1761213600000,
      engine: 'blink',
      in_app_browser: null,
    },
    capabilities: {
      secure_context: true,
      device_motion: true,
      device_orientation: true,
      permission_api: true,
      permission_state: 'granted',
      generic_relative_orientation: true,
      wake_lock: true,
      vibrate: true,
      share_files: true,
      measured: {
        duration_ms: 2000,
        motion_hz: 60,
        orientation_hz: 60,
        quaternion_hz: null,
        motion_dt_ms_median: 16.7,
        motion_dt_ms_p95: 18,
        acc_available: true,
        accg_available: true,
        rotation_available: true,
        interval_raw: 16,
      },
    },
    calibration: null,
    protocol: {
      preset: 'quick',
      protocol_version: '1.0.0',
      seed: 12345,
      posture: 'standing',
      first_hand: 'right',
      n_planned_trials: 2,
      estimated_minutes: 2,
      custom: null,
      plan: [
        {
          block_index: 0,
          kind: 'main',
          hand: 'right',
          orientation: 'upright_toward',
          trials: [
            { trial_index: 0, block_index: 0, trial_in_block: 0, condition: makeCondition() },
            { trial_index: 1, block_index: 0, trial_in_block: 1, condition: makeCondition({ speed: 'fast' }) },
          ],
        },
      ],
    },
    settings: { language: 'en', audio: true, speech: false, segmentation_params: { ...DEFAULT_SEGMENTATION_PARAMS } },
    progress: { next_trial_index: 0, n_attempts: 0, n_accepted: 0 },
    consent: { version: '2026-10-v1', accepted_at: T0 },
    timezone_offset_min: -180,
    simulated: false,
    notes: null,
    ...over,
  };
}

export function makeTrial(over: Partial<TrialRecord> = {}): TrialRecord {
  return {
    record_type: 'trial',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    trial_id: 'trial-0001',
    session_id: 'sess-0001',
    participant_id: 'P-ABCDEF',
    study_code: 'nose-touch-v1',
    created_at: T0,
    block_index: 0,
    trial_index: 0,
    trial_in_block: 0,
    attempt: 1,
    simulated: false,
    condition: makeCondition(),
    outcome: { status: 'complete', accepted: true, participant_flag: null, note: null },
    timing: {
      t0_perf_ms: 1000,
      t0_epoch_ms: 1761213600000,
      still_ms: 500,
      go_ms: 900,
      onset_ms: 1100,
      end_ms: 1900,
      end_detected_ms: 2200,
      stop_ms: 3000,
      first_touch_ms: null,
      foreperiod_ms: 400,
      false_starts: 0,
    },
    segmentation: {
      method: 'online_v1',
      params: { ...DEFAULT_SEGMENTATION_PARAMS },
      acc_source: 'linear',
      onset_index: 1,
      end_index: 2,
      duration_ms: 800,
      peak_gyro_dps: 240.5,
      peak_acc_ms2: 6.25,
      baseline: { n: 30, gyro_mean: 1.2, gyro_sd: 0.4, acc_mean: 0.05, acc_sd: 0.02, gravity: [0.1, 9.79, 0.3] },
    },
    detected: { rotation_start: 'upright', rotation_end: 'upright', tilt_change_deg: 42.5, orientation_match: true },
    quality: {
      n_motion: 3,
      n_orientation: 2,
      n_quaternion: 0,
      motion_hz: 60,
      motion_dt_ms_median: 16.7,
      motion_dt_ms_max: 16.8,
      gaps_over_50ms: 0,
      acc_available: true,
      rotation_available: true,
      hidden_during: false,
      warnings: [],
    },
    screen: { type_start: 'portrait-primary', angle_start: 0, type_end: 'portrait-primary', angle_end: 0 },
    streams: {
      motion: {
        t: [0, 16.7, 33.4],
        ts: [0, 16.7, 33.4],
        acc_x: [0.01, 0.5, -0.2],
        acc_y: [0.02, 1.5, 0.3],
        acc_z: [-0.01, 0.8, null],
        accg_x: [0.1, 0.6, -0.1],
        accg_y: [9.8, 11.2, 9.9],
        accg_z: [0.3, 1.1, 0.2],
        rr_alpha: [0.5, 120.25, 10],
        rr_beta: [-0.2, 30, 2],
        rr_gamma: [0.1, -15.5, 1],
        interval_raw: 16,
        ts_first_raw: 12345.6,
      },
      orientation: {
        t: [0, 16.7],
        ts: [0, 16.7],
        alpha: [10, 11],
        beta: [80, 75],
        gamma: [1, 2],
        compass: [null, null],
        absolute: false,
        ts_first_raw: 12345.7,
      },
      quaternion: null,
      touches: [],
      cues: [{ t: 900, kind: 'go' }],
    },
    ...over,
  };
}

export function makeBundle(over: Partial<ExportBundle> = {}): ExportBundle {
  const session = makeSession();
  return {
    format: 'nose-touch-export',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    exported_at: T0,
    participants: [session.participant],
    sessions: [session],
    trials: [makeTrial()],
    ...over,
  };
}

export function makeEnvelope(
  kind: 'session' | 'trial',
  record: SessionRecord | TrialRecord,
  studyKey: string | null = null,
): UploadEnvelope {
  return { kind, study_key: studyKey, sent_at: T0, record };
}

export const APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbTEST/exec';

export function appsScriptCfg(over: Partial<UploadConfig> = {}): UploadConfig {
  return { mode: 'apps_script', endpoint: APPS_SCRIPT_URL, study_key: 'k-123', ...over };
}

/** Distinct record ids for tests: `${prefix}-0001`, ... */
export function ids(prefix: string, n: number): string[] {
  return Array.from({ length: n }, (_, i) => `${prefix}-${String(i + 1).padStart(4, '0')}`);
}
