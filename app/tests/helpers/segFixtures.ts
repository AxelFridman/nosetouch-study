/**
 * Shared fixtures for the segmentation / recorder / controller tests: parameters, ids, conditions,
 * a recording cue sink and small numeric helpers.
 */
import { DEFAULT_SEGMENTATION_PARAMS } from '../../src/core/constants';
import type { RelMotionSample } from '../../src/core/segmentation';
import type {
  CueKind,
  CueSink,
  Speed,
  SegmentationParams,
  TrialCondition,
  TrialIds,
  TrialRecord,
} from '../../src/core/types';

/**
 * Per-speed overrides, copied from the protocol.resolveParams doc so these tests do not depend on
 * protocol.ts (written by another agent): slow → lower thresholds, longer end sustain/max movement;
 * fast → shorter onset sustain.
 */
export function segParamsForSpeed(speed: Speed, base: SegmentationParams = DEFAULT_SEGMENTATION_PARAMS): SegmentationParams {
  if (speed === 'slow') {
    return {
      ...base,
      onset_gyro_min_dps: 25,
      onset_acc_min_ms2: 0.55,
      end_gyro_dps: 10,
      end_acc_ms2: 0.25,
      end_sustain_ms: 450,
      max_movement_ms: 12000,
    };
  }
  if (speed === 'fast') return { ...base, onset_sustain_ms: 40 };
  return { ...base };
}

export const SEG_IDS: TrialIds = {
  trial_id: 'trial-0001',
  session_id: 'session-0001',
  participant_id: 'P-ABCDE2',
  study_code: 'nose-touch-v1',
  block_index: 0,
  trial_index: 3,
  trial_in_block: 3,
  attempt: 1,
};

export function segCondition(overrides: Partial<TrialCondition> = {}): TrialCondition {
  return {
    hand: 'right',
    orientation: 'upright_toward',
    speed: 'natural',
    start_position: 'natural',
    variant: 'standard',
    posture: 'standing',
    rep: 0,
    ...overrides,
  };
}

export interface RecordingCueSink extends CueSink {
  played: CueKind[];
}

export function segCueSink(): RecordingCueSink {
  const played: CueKind[] = [];
  return {
    played,
    play(kind: CueKind): void {
      played.push(kind);
    },
  };
}

/** Converts a stored motion stream back into segmenter input (what segment.py does in Python). */
export function storedMotionSamples(rec: TrialRecord): RelMotionSample[] {
  const m = rec.streams.motion;
  const vec = (x: number | null, y: number | null, z: number | null): [number, number, number] | null =>
    x !== null && y !== null && z !== null ? [x, y, z] : null;
  return m.t.map((t, i) => {
    const a = m.rr_alpha[i];
    const b = m.rr_beta[i];
    const g = m.rr_gamma[i];
    return {
      t,
      ts: m.ts[i],
      acc: vec(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
      accg: vec(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
      rr: a !== null && b !== null && g !== null ? { alpha: a, beta: b, gamma: g } : null,
      interval: null,
      index: i,
    };
  });
}

/** Every number anywhere inside `v` is finite (JSON would silently turn NaN/Infinity into null). */
export function allNumbersFinite(v: unknown, path = '$'): string[] {
  if (typeof v === 'number') return Number.isFinite(v) ? [] : [path];
  if (Array.isArray(v)) return v.flatMap((x, i) => allNumbersFinite(x, `${path}[${i}]`));
  if (v && typeof v === 'object') {
    return Object.entries(v as Record<string, unknown>).flatMap(([k, x]) => allNumbersFinite(x, `${path}.${k}`));
  }
  return [];
}

export function fmt(n: number | null): string {
  return n === null ? 'null' : n.toFixed(1);
}
