/**
 * Builds schema-valid TrialRecords straight from simulator output (no TrialController needed), for
 * tests of mirrorTrial and other record-level code. Values other than the streams are plausible
 * constants; the streams are copied unrounded.
 */
import { APP_VERSION, DEFAULT_SEGMENTATION_PARAMS, SCHEMA_VERSION } from '../../src/core/constants';
import type { TrialCondition, TrialRecord } from '../../src/core/types';
import { START_T, type Collected } from './simHarness';

export function trialFromCollected(c: Collected, cond: TrialCondition, frequencyHz = 60): TrialRecord {
  const m = c.motion;
  const o = c.orientation;
  const q = c.quaternion;
  const tsFirst = m.length ? m[0].ts : null;
  const tsFirstO = o.length ? o[0].ts : null;
  const rel = (v: number | null, base: number | null): number | null => (v === null || base === null ? null : v - base);
  const go = c.source.truth.go_t;
  const t0 = START_T;
  const last = m.length ? m[m.length - 1].t - t0 : 0;
  return {
    record_type: 'trial',
    schema_version: SCHEMA_VERSION,
    app_version: APP_VERSION,
    trial_id: 'trial-fixture-1',
    session_id: 'session-fixture-1',
    participant_id: 'P-ABCDEF',
    study_code: 'test-study',
    created_at: '2026-10-23T12:00:00.000Z',
    block_index: 0,
    trial_index: 3,
    trial_in_block: 3,
    attempt: 1,
    simulated: true,
    condition: { ...cond },
    outcome: { status: 'complete', accepted: true, participant_flag: null, note: null },
    timing: {
      t0_perf_ms: t0,
      t0_epoch_ms: 1_700_000_000_000 + t0,
      still_ms: 500,
      go_ms: go === null ? null : go - t0,
      onset_ms: c.source.truth.onset_t === null ? null : c.source.truth.onset_t - t0,
      end_ms: c.source.truth.end_t === null ? null : c.source.truth.end_t - t0,
      end_detected_ms: null,
      stop_ms: last,
      first_touch_ms: null,
      foreperiod_ms: 700,
      false_starts: 0,
    },
    segmentation: {
      method: 'online_v1',
      params: { ...DEFAULT_SEGMENTATION_PARAMS },
      acc_source: 'linear',
      onset_index: 10,
      end_index: 60,
      duration_ms: 900,
      peak_gyro_dps: 120,
      peak_acc_ms2: 4,
      baseline: { n: 30, gyro_mean: 2, gyro_sd: 0.5, acc_mean: 0.1, acc_sd: 0.02, gravity: [1.5, 9.6, 0.4] },
    },
    detected: { rotation_start: 'top_left', rotation_end: 'upright', tilt_change_deg: 12.5, orientation_match: true },
    quality: {
      n_motion: m.length,
      n_orientation: o.length,
      n_quaternion: q.length,
      motion_hz: frequencyHz,
      motion_dt_ms_median: 1000 / frequencyHz,
      motion_dt_ms_max: 40,
      gaps_over_50ms: 0,
      acc_available: m.some((s) => s.acc !== null),
      rotation_available: m.some((s) => s.rr !== null),
      hidden_during: false,
      warnings: [],
    },
    screen: { type_start: 'landscape-primary', angle_start: 90, type_end: 'portrait-primary', angle_end: 0 },
    streams: {
      motion: {
        t: m.map((s) => s.t - t0),
        ts: m.map((s) => rel(s.ts, tsFirst)),
        acc_x: m.map((s) => (s.acc ? s.acc[0] : null)),
        acc_y: m.map((s) => (s.acc ? s.acc[1] : null)),
        acc_z: m.map((s) => (s.acc ? s.acc[2] : null)),
        accg_x: m.map((s) => (s.accg ? s.accg[0] : null)),
        accg_y: m.map((s) => (s.accg ? s.accg[1] : null)),
        accg_z: m.map((s) => (s.accg ? s.accg[2] : null)),
        rr_alpha: m.map((s) => (s.rr ? s.rr.alpha : null)),
        rr_beta: m.map((s) => (s.rr ? s.rr.beta : null)),
        rr_gamma: m.map((s) => (s.rr ? s.rr.gamma : null)),
        interval_raw: m.length ? m[0].interval : null,
        ts_first_raw: tsFirst,
      },
      orientation: o.length
        ? {
            t: o.map((s) => s.t - t0),
            ts: o.map((s) => rel(s.ts, tsFirstO)),
            alpha: o.map((s) => s.alpha),
            beta: o.map((s) => s.beta),
            gamma: o.map((s) => s.gamma),
            compass: o.map((s) => s.compass),
            absolute: o[0].absolute,
            ts_first_raw: tsFirstO,
          }
        : null,
      quaternion: q.length
        ? {
            t: q.map((s) => s.t - t0),
            x: q.map((s) => s.q[0]),
            y: q.map((s) => s.q[1]),
            z: q.map((s) => s.q[2]),
            w: q.map((s) => s.q[3]),
            sensor: q[0].sensor,
            frequency_hz: frequencyHz,
          }
        : null,
      touches: [{ t: 1200, type: 'start', n: 1, x: 100, y: 200 }],
      cues: go === null ? [] : [{ t: go - t0, kind: 'go' }],
    },
  };
}
