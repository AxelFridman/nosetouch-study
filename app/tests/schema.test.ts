/**
 * The data contract schema/records.schema.json, version 1.2.0: every addition is optional, so records
 * written by older app versions (the Android pilot, schema 1.1.0) stay valid, while v2 records with
 * every new field validate and a record missing a required key is still rejected.
 */
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { DEFAULT_SEGMENTATION_PARAMS, PROTOCOL_VERSION, SCHEMA_VERSION } from '../src/core/constants';
import { buildPlan, estimateMinutes, flattenPlan } from '../src/core/protocol';
import type { SegmentationParams, SessionRecord, TrialRecord } from '../src/core/types';
import { T0, makeBundle, makeCondition, makeEnvelope, makeSession, makeTrial } from './helpers/dataFixtures';
import { expectValid, recordsSchema, validate } from './helpers/schema';

/** The pilot export (gitignored, personal data): at the repository root or in analysis/data/real/. */
const PILOT = 'nosetouch_P-VPSDL4_20261024-0159.json';
const pilotPath = [`../../${PILOT}`, `../../analysis/data/real/${PILOT}`]
  .map((p) => fileURLToPath(new URL(p, import.meta.url)))
  .find((p) => existsSync(p));

/** Segmentation params as schema 1.1 apps wrote them (no return_cap_ms). */
function v11Params(): SegmentationParams {
  const { return_cap_ms: _dropped, ...rest } = DEFAULT_SEGMENTATION_PARAMS;
  return rest;
}

/** A session as the v2 app writes it: street plan, implicit calibration, review mode, status ended_early. */
function v12Session(): SessionRecord {
  const plan = buildPlan({ preset: 'street', seed: 7, posture: 'standing' });
  return makeSession({
    status: 'ended_early',
    ended_at: T0,
    calibration: {
      method: 'implicit_v1',
      performed_at: T0,
      flat: null,
      upright: null,
      gravity_sign: null,
      gravity_magnitude: null,
      upright_check: null,
      gyro_bias_dps: null,
      notes: [],
      axis_check: null,
    },
    protocol: {
      preset: 'street',
      protocol_version: PROTOCOL_VERSION,
      seed: 7,
      posture: 'standing',
      first_hand: 'left',
      n_planned_trials: flattenPlan(plan).length,
      estimated_minutes: estimateMinutes(plan),
      custom: null,
      plan,
    },
    settings: { language: 'he', audio: true, speech: true, segmentation_params: { ...DEFAULT_SEGMENTATION_PARAMS }, review_mode: 'auto' },
    consent: { version: '2026-10-v3', accepted_at: T0 },
  });
}

/** A trial with every optional 1.2 field set. */
function v12Trial(): TrialRecord {
  const base = makeTrial();
  return {
    ...base,
    condition: makeCondition({ variant: 'tilted_touch', hand_source: 'reported' }),
    timing: { ...base.timing, return_onset_ms: 2650.5, return_end_ms: null },
    segmentation: { ...base.segmentation, params: { ...DEFAULT_SEGMENTATION_PARAMS } },
    quality: { ...base.quality, n_quaternion_abs: 2, n_orientation_abs: 1 },
    streams: {
      ...base.streams,
      orientation: { ...base.streams.orientation!, compass_accuracy: [null, 15] },
      touches: [
        { t: 1500, type: 'start', n: 1, x: 120.5, y: 80.25, radius_x: 11.5, radius_y: 9, force: 0, rotation_angle: 12.5, x_norm: 0.29, y_norm: 0.09 },
        { t: 1620, type: 'end', n: 0, x: 121, y: 80, radius_x: null, radius_y: null, force: null, rotation_angle: null, x_norm: null, y_norm: null },
      ],
      quaternion_abs: { t: [0, 16.7], x: [0, 0.1], y: [0, 0], z: [0, 0], w: [1, 0.995], sensor: 'AbsoluteOrientationSensor', frequency_hz: 60 },
      orientation_abs: { t: [0], ts: [0], alpha: [123.4], beta: [80], gamma: [1], compass: [null], absolute: true, ts_first_raw: 12345.8 },
      screen_events: [
        { t: 1700, type: 'landscape-primary', angle: 90 },
        { t: 2500, type: null, angle: null },
      ],
      magnetometer: { t: [0, 100], x: [12.5, null], y: [-30, -29.5], z: [40, 41], frequency_hz: 10 },
    },
  };
}

/** A deep copy of `rec` with the value at `path` replaced (`undefined` deletes the key). */
function withValue<T>(rec: T, path: (string | number)[], value: unknown): T {
  const copy = structuredClone(rec) as Record<string | number, unknown>;
  let node = copy;
  for (const k of path.slice(0, -1)) node = node[k] as Record<string | number, unknown>;
  const last = path[path.length - 1];
  if (value === undefined) delete node[last];
  else node[last] = value;
  return copy as T;
}

describe('schema 1.2.0', () => {
  it('is the version the app writes', () => {
    expect(SCHEMA_VERSION).toBe('1.2.0');
    expect(new RegExp(recordsSchema.$defs.SessionRecord.properties.schema_version.pattern).test(SCHEMA_VERSION)).toBe(true);
  });

  it.skipIf(!pilotPath)('the v1 pilot export (schema 1.1.0, Android) still validates, record by record', () => {
    const bundle = JSON.parse(readFileSync(pilotPath as string, 'utf8'));
    expect(bundle.schema_version).toBe('1.1.0');
    expect(bundle.trials.length).toBeGreaterThan(0);
    expectValid('ExportBundle', bundle);
    for (const p of bundle.participants) expectValid('ParticipantProfile', p);
    for (const s of bundle.sessions) {
      expectValid('SessionRecord', s);
      expectValid('UploadEnvelope', makeEnvelope('session', s));
    }
    for (const t of bundle.trials) {
      expectValid('TrialRecord', t);
      expectValid('UploadEnvelope', makeEnvelope('trial', t));
    }
  });

  it('records without any 1.2 field (schema 1.1 shape) still validate', () => {
    const session = makeSession({ settings: { language: 'en', audio: true, speech: false, segmentation_params: v11Params() } });
    const trial = makeTrial({ segmentation: { ...makeTrial().segmentation, params: v11Params() } });
    expect(session.protocol.plan[0]).not.toHaveProperty('tier');
    expect(trial.condition).not.toHaveProperty('hand_source');
    expectValid('SessionRecord', session);
    expectValid('TrialRecord', trial);
    expectValid('ExportBundle', makeBundle({ sessions: [session], trials: [trial] }));
  });

  it('a session and a trial using every new optional field validate', () => {
    const session = v12Session();
    const trial = v12Trial();
    expect(session.protocol.plan.some((b) => b.kind === 'choice' && b.tier === 0)).toBe(true);
    expect(session.protocol.plan.some((b) => b.optional === true)).toBe(true);
    expect(session.settings.segmentation_params.return_cap_ms).toBe(3000);
    expectValid('SessionRecord', session);
    expectValid('TrialRecord', trial);
    expectValid('UploadEnvelope', makeEnvelope('session', session, 'k'));
    expectValid('UploadEnvelope', makeEnvelope('trial', trial, 'k'));
    expectValid('ExportBundle', makeBundle({ sessions: [session], participants: [session.participant], trials: [trial] }));
    // the optional streams may also be null / empty
    const empty = withValue(withValue(withValue(withValue(trial, ['streams', 'quaternion_abs'], null), ['streams', 'orientation_abs'], null), ['streams', 'magnetometer'], null), ['streams', 'screen_events'], []);
    expectValid('TrialRecord', empty);
  });

  it('every new property is optional (in no `required` list), so old records stay valid', () => {
    const defs = recordsSchema.$defs;
    const objects: Record<string, { properties: Record<string, unknown>; required: string[] }> = {
      TrialCondition: defs.TrialCondition,
      SegmentationParams: defs.SegmentationParams,
      CalibrationResult: defs.CalibrationResult,
      PlannedBlock: defs.PlannedBlock,
      OrientationStream: defs.OrientationStream,
      TouchRecord: defs.TouchRecord,
      'SessionRecord.settings': defs.SessionRecord.properties.settings,
      'TrialRecord.timing': defs.TrialRecord.properties.timing,
      'TrialRecord.quality': defs.TrialRecord.properties.quality,
      'TrialRecord.streams': defs.TrialRecord.properties.streams,
    };
    const added: Record<string, string[]> = {
      TrialCondition: ['hand_source'],
      SegmentationParams: ['return_cap_ms'],
      CalibrationResult: ['method'],
      PlannedBlock: ['tier', 'optional'],
      OrientationStream: ['compass_accuracy'],
      TouchRecord: ['radius_x', 'radius_y', 'force', 'rotation_angle', 'x_norm', 'y_norm'],
      'SessionRecord.settings': ['review_mode'],
      'TrialRecord.timing': ['return_onset_ms', 'return_end_ms'],
      'TrialRecord.quality': ['n_quaternion_abs', 'n_orientation_abs'],
      'TrialRecord.streams': ['quaternion_abs', 'orientation_abs', 'screen_events', 'magnetometer'],
    };
    for (const [name, props] of Object.entries(added)) {
      for (const p of props) {
        expect(objects[name].properties, `${name}.${p}`).toHaveProperty(p);
        expect(objects[name].required, `${name}.${p}`).not.toContain(p);
      }
    }
  });

  it('accepts the new enum values', () => {
    const defs = recordsSchema.$defs;
    expect(defs.ProtocolPreset.enum).toContain('street');
    expect(defs.Variant.enum).toContain('tilted_touch');
    expect(defs.PlannedBlock.properties.kind.enum).toContain('choice');
    expect(defs.SessionRecord.properties.status.enum).toContain('ended_early');
    for (const status of ['in_progress', 'completed', 'abandoned', 'ended_early'] as const) {
      expectValid('SessionRecord', makeSession({ status }));
    }
    const s = makeSession();
    expectValid('CalibrationResult', { ...v12Session().calibration, method: 'guided_v1' });
    expectValid('SessionRecord', withValue(s, ['settings', 'review_mode'], 'tap'));
    expectValid('TrialRecord', withValue(makeTrial(), ['condition', 'hand_source'], 'instructed'));
  });

  it('still rejects a record that misses a required key', () => {
    const session = v12Session();
    const trial = v12Trial();
    const missing: [string, (string | number)[]][] = [
      ['SessionRecord', ['status']],
      ['SessionRecord', ['participant', 'ehi_sf']], // dropped questions must be written as null, not omitted
      ['SessionRecord', ['calibration', 'notes']],
      ['SessionRecord', ['settings', 'segmentation_params', 'post_roll_ms']],
      ['SessionRecord', ['protocol', 'plan', 0, 'kind']],
      ['TrialRecord', ['timing', 'false_starts']],
      ['TrialRecord', ['streams', 'cues']],
      ['TrialRecord', ['condition', 'rep']],
      ['TrialRecord', ['streams', 'touches', 0, 'n']],
      ['TrialRecord', ['streams', 'screen_events', 0, 'angle']],
      ['TrialRecord', ['streams', 'magnetometer', 'frequency_hz']],
      ['TrialRecord', ['streams', 'quaternion_abs', 'sensor']],
    ];
    for (const [def, path] of missing) {
      const rec = def === 'SessionRecord' ? session : trial;
      expect(validate(def as 'SessionRecord' | 'TrialRecord', withValue(rec, path, undefined)), path.join('.')).not.toEqual([]);
    }
  });

  it('rejects values outside the new enums and ranges', () => {
    const session = v12Session();
    const trial = v12Trial();
    const bad: [string, (string | number)[], unknown][] = [
      ['SessionRecord', ['status'], 'quit'],
      ['SessionRecord', ['settings', 'review_mode'], 'manual'],
      ['SessionRecord', ['calibration', 'method'], 'table'],
      ['SessionRecord', ['protocol', 'preset'], 'street2'],
      ['SessionRecord', ['protocol', 'plan', 0, 'kind'], 'bonus'],
      ['SessionRecord', ['protocol', 'plan', 0, 'tier'], 10],
      ['SessionRecord', ['protocol', 'plan', 0, 'tier'], -1],
      ['SessionRecord', ['protocol', 'plan', 0, 'tier'], 1.5],
      ['SessionRecord', ['protocol', 'plan', 0, 'optional'], 'yes'],
      ['SessionRecord', ['settings', 'segmentation_params', 'return_cap_ms'], -1],
      ['TrialRecord', ['condition', 'hand_source'], 'guessed'],
      ['TrialRecord', ['condition', 'variant'], 'tilted'],
      ['TrialRecord', ['timing', 'return_onset_ms'], '2650'],
      ['TrialRecord', ['quality', 'n_quaternion_abs'], -1],
      ['TrialRecord', ['streams', 'touches', 0, 'force'], 'hard'],
      ['TrialRecord', ['streams', 'touches', 0, 'pressure'], 0.5], // unknown keys stay forbidden
      ['TrialRecord', ['streams', 'screen_events'], null],
      ['TrialRecord', ['streams', 'magnetometer', 'x'], [1, 'a']],
      ['TrialRecord', ['streams', 'orientation', 'compass_accuracy'], 15],
    ];
    for (const [def, path, value] of bad) {
      const rec = def === 'SessionRecord' ? session : trial;
      expect(validate(def as 'SessionRecord' | 'TrialRecord', withValue(rec, path, value)), `${path.join('.')} = ${JSON.stringify(value)}`).not.toEqual([]);
    }
  });
});
