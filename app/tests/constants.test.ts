import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  ALL_ORIENTATIONS,
  APP_VERSION,
  COLLECTED_ORIENTATIONS,
  CONSENT_VERSION_DEFAULT,
  CORE_ORIENTATIONS,
  DEFAULT_SEGMENTATION_PARAMS,
  ODD_VARIANTS,
  ORIENTATIONS,
  PROTOCOL_VERSION,
  SCHEMA_VERSION,
} from '../src/core/constants';
import { recordsSchema } from './helpers/schema';

describe('constants', () => {
  it('APP_VERSION matches package.json', () => {
    const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    expect(APP_VERSION).toBe(pkg.version);
  });

  it('SCHEMA_VERSION is accepted by the schema pattern', () => {
    const pattern = new RegExp(recordsSchema.$defs.SessionRecord.properties.schema_version.pattern);
    expect(pattern.test(SCHEMA_VERSION)).toBe(true);
  });

  it('v2 versions: app 0.2.1, schema 1.2.0, protocol 2.0.0, consent 2026-10-v3', () => {
    expect([APP_VERSION, SCHEMA_VERSION, PROTOCOL_VERSION, CONSENT_VERSION_DEFAULT]).toEqual(['0.2.1', '1.2.0', '2.0.0', '2026-10-v3']);
  });

  it('orientation mirror map is an involution and matches the schema enum', () => {
    const ids = Object.keys(ORIENTATIONS);
    expect(new Set(ids)).toEqual(new Set(recordsSchema.$defs.OrientationId.enum));
    for (const id of ids) {
      const m = ORIENTATIONS[id as keyof typeof ORIENTATIONS].mirror;
      expect(ORIENTATIONS[m].mirror).toBe(id);
    }
  });

  it('the collected and core orientations never face the screen away, and are closed under mirroring', () => {
    expect(COLLECTED_ORIENTATIONS.every((o) => ORIENTATIONS[o].screen !== 'away')).toBe(true);
    expect(ALL_ORIENTATIONS.filter((o) => !COLLECTED_ORIENTATIONS.includes(o))).toEqual(['upright_away', 'upside_down_away', 'top_left_away', 'top_right_away']);
    for (const list of [COLLECTED_ORIENTATIONS, CORE_ORIENTATIONS]) {
      for (const o of list) expect(list).toContain(ORIENTATIONS[o].mirror);
    }
    expect(CORE_ORIENTATIONS.every((o) => COLLECTED_ORIENTATIONS.includes(o))).toBe(true);
  });

  it('ODD_VARIANTS are the schema variants except standard (tilted_touch included)', () => {
    expect([...ODD_VARIANTS].sort()).toEqual(recordsSchema.$defs.Variant.enum.filter((v: string) => v !== 'standard').sort());
    expect(ODD_VARIANTS).toContain('tilted_touch');
  });

  it('pins the protocol 2.0.0 segmentation parameters (spec section 4; mirrored in analysis/motionlab/conventions.py)', () => {
    expect(DEFAULT_SEGMENTATION_PARAMS).toEqual({
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
      post_roll_ms: 500,
      return_cap_ms: 3000,
    });
    // onset floors above the still thresholds: a still sway can never be an onset
    expect(DEFAULT_SEGMENTATION_PARAMS.onset_gyro_min_dps).toBeGreaterThan(DEFAULT_SEGMENTATION_PARAMS.still_gyro_dps);
    expect(DEFAULT_SEGMENTATION_PARAMS.onset_acc_min_ms2).toBeGreaterThan(DEFAULT_SEGMENTATION_PARAMS.still_acc_ms2);
    // ≥ 0.8 s of verified stillness before GO (offline baseline 500 ms + margin)
    expect(DEFAULT_SEGMENTATION_PARAMS.still_hold_ms + DEFAULT_SEGMENTATION_PARAMS.foreperiod_min_ms).toBeGreaterThanOrEqual(800);
  });
});
