import { describe, expect, it } from 'vitest';
import { ROUNDING } from '../src/core/constants';
import {
  CLIP_ACC_MS2,
  CLIP_GYRO_DPS,
  LOW_MOTION_HZ,
  StreamRecorder,
  computeQuality,
  rateStats,
  roundValue,
} from '../src/core/recorder';
import type { MotionSample } from '../src/core/types';
import { expectEqualColumnLengths, validate, type SchemaDef } from './helpers/schema';
import { allNumbersFinite } from './helpers/segFixtures';

const T0 = 123456.789;

function motion(t: number, over: Partial<MotionSample> = {}): MotionSample {
  return {
    t,
    ts: t - 3.5,
    acc: [0.123456, -0.0000449, 9.99995],
    accg: [1.23456789, 9.80665, -0.00005],
    rr: { alpha: 1.23456, beta: -2.0005, gamma: 300.12345 },
    interval: 16,
    ...over,
  };
}

/** The stream defs are not in the SchemaDef union; validate() resolves any $defs name at runtime. */
function streamErrors(def: string, v: unknown): string[] {
  return validate(def as SchemaDef, v);
}

describe('roundValue', () => {
  it('rounds halves away from zero, never returns -0, NaN or Infinity', () => {
    expect(roundValue(1.005, 2)).toBe(1); // the double nearest 1.005 is 1.00499999999999989…
    expect(roundValue(2.5, 0)).toBe(3);
    expect(roundValue(-2.5, 0)).toBe(-3);
    expect(roundValue(0.125, 2)).toBe(0.13);
    expect(roundValue(-0.125, 2)).toBe(-0.13);
    expect(Object.is(roundValue(-0.0001, 2), 0)).toBe(true);
    expect(roundValue(Number.NaN, 2)).toBeNull();
    expect(roundValue(Number.POSITIVE_INFINITY, 2)).toBeNull();
    expect(roundValue(null, 2)).toBeNull();
    expect(roundValue(undefined, 2)).toBeNull();
    expect(roundValue(1.7e308, 6)).toBe(1.7e308); // no overflow to Infinity
    expect(roundValue(1234567.123456789, 4)).toBe(1234567.1235);
  });
});

describe('StreamRecorder', () => {
  it('stores times relative to t0 and rounds every column per ROUNDING', () => {
    const r = new StreamRecorder(T0);
    expect(r.addMotion(motion(T0 + 10.123456))).toBe(0);
    expect(r.addMotion(motion(T0 + 26.98765))).toBe(1);
    r.addOrientation({ t: T0 + 11.11111, ts: 500.123456, alpha: 359.99987, beta: -45.12345, gamma: 0.00049, absolute: true, compass: 12.34567 });
    r.addQuaternion({ t: T0 + 12.5, q: [0.12345678, -0.5, 0.70710678, 0.4999999], sensor: 'RelativeOrientationSensor' }, 60);
    const s = r.build();
    expect(s.motion.t).toEqual([10.12, 26.99]);
    expect(s.motion.acc_x[0]).toBe(0.1235);
    expect(s.motion.acc_y[0]).toBe(0); // -0.0000449 → 0 (not -0)
    expect(Object.is(s.motion.acc_y[0], 0)).toBe(true);
    expect(s.motion.acc_z[0]).toBe(10); // 9.99995 rounds up at 4 decimals
    expect(s.motion.accg_x[0]).toBe(1.2346);
    expect(s.motion.accg_y[0]).toBe(9.8067);
    expect(s.motion.accg_z[0]).toBe(-0.0001);
    expect(s.motion.rr_alpha[0]).toBe(1.235);
    expect(s.motion.rr_beta[0]).toBe(-2.001);
    expect(s.motion.rr_gamma[0]).toBe(300.123);
    expect(s.orientation).toMatchObject({ t: [11.11], alpha: [360], beta: [-45.123], gamma: [0], compass: [12.346], absolute: true });
    expect(s.quaternion).toMatchObject({
      t: [12.5],
      x: [0.123457],
      y: [-0.5],
      z: [0.707107],
      w: [0.5],
      sensor: 'RelativeOrientationSensor',
      frequency_hz: 60,
    });
    expect(ROUNDING).toEqual({ time: 2, acc: 4, rr: 3, angle: 3, quat: 6 });
  });

  it('ts is relative to the first finite raw timeStamp; ts_first_raw and interval_raw are kept raw', () => {
    const r = new StreamRecorder(1000);
    r.addMotion(motion(1000, { ts: null, interval: null }));
    r.addMotion(motion(1016.6, { ts: 98765.4321, interval: 0.016667 }));
    r.addMotion(motion(1033.3, { ts: 98782.1009, interval: 16 }));
    r.addMotion(motion(1050, { ts: Number.NaN }));
    const m = r.build().motion;
    expect(m.ts).toEqual([null, 0, 16.67, null]);
    expect(m.ts_first_raw).toBe(98765.4321);
    expect(m.interval_raw).toBe(0.016667); // the first finite one, unconverted (iOS reports seconds)
  });

  it('turns every missing / non-finite value into null and keeps the columns aligned', () => {
    const r = new StreamRecorder(0);
    const bad = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
    r.addMotion(motion(1, { acc: null, accg: null, rr: null, ts: null, interval: null }));
    r.addMotion(motion(2, { acc: [bad[0], 1, 2], accg: [3, bad[1], 4], rr: { alpha: bad[2], beta: 1, gamma: 2 } }));
    r.addMotion(motion(3, { acc: undefined as unknown as null, rr: { alpha: 1, beta: 2 } as never }));
    r.addMotion(motion(4, { interval: bad[0], ts: bad[1] }));
    r.addOrientation({ t: 5, ts: null, alpha: null, beta: Number.NaN, gamma: 3, absolute: null, compass: Number.NaN });
    r.addOrientation({ t: Number.NaN, ts: 1, alpha: 1, beta: 1, gamma: 1, absolute: false, compass: null }); // skipped
    r.addQuaternion({ t: 6, q: [0, 0, Number.NaN, 1], sensor: 'AbsoluteOrientationSensor' }, 60); // skipped
    r.addQuaternion({ t: 7, q: [0, 0, 0, 1], sensor: 'AbsoluteOrientationSensor' }, Number.NaN);
    r.addTouch(Number.NaN, 'start', 1, 1, 1); // skipped
    r.addTouch(8, 'start', Number.NaN, Number.NaN, null);
    r.addCue(Number.POSITIVE_INFINITY, 'go'); // skipped
    r.addCue(9, 'go');
    const s = r.build();
    expect(allNumbersFinite(s)).toEqual([]);
    expect(JSON.parse(JSON.stringify(s))).toEqual(s); // nothing JSON would mangle
    expectEqualColumnLengths(s.motion as unknown as Record<string, unknown>);
    expectEqualColumnLengths(s.orientation as unknown as Record<string, unknown>);
    expectEqualColumnLengths(s.quaternion as unknown as Record<string, unknown>);
    expect(s.motion.t.length).toBe(4);
    expect(s.motion.acc_x).toEqual([null, null, null, 0.1235]);
    expect(s.motion.acc_y).toEqual([null, 1, null, 0]);
    expect(s.motion.accg_y[1]).toBeNull();
    expect(s.motion.rr_alpha[1]).toBeNull();
    expect(s.motion.rr_gamma[2]).toBeNull();
    expect(s.motion.interval_raw).toBe(16);
    expect(s.orientation).toMatchObject({ t: [5], alpha: [null], beta: [null], gamma: [3], compass: [null], absolute: null });
    expect(s.quaternion).toMatchObject({ t: [7], sensor: 'AbsoluteOrientationSensor', frequency_hz: 0 });
    expect(s.touches).toEqual([{ t: 8, type: 'start', n: 0, x: null, y: null }]);
    expect(s.cues).toEqual([{ t: 9, kind: 'go' }]);
    for (const [def, v] of [
      ['MotionStream', s.motion],
      ['OrientationStream', s.orientation],
      ['QuaternionStream', s.quaternion],
    ] as const) {
      expect(streamErrors(def, v), def).toEqual([]);
    }
  });

  it('keeps a motion slot (previous t) for a non-finite time so indices stay aligned', () => {
    const r = new StreamRecorder(100);
    r.addMotion(motion(Number.NaN));
    r.addMotion(motion(110));
    r.addMotion(motion(Number.POSITIVE_INFINITY));
    expect(r.build().motion.t).toEqual([0, 10, 10]);
  });

  it('addMotionRel returns exactly the stored values (what the segmenter is fed)', () => {
    const r = new StreamRecorder(T0);
    for (let i = 0; i < 5; i++) {
      const rel = r.addMotionRel(
        motion(T0 + i * 16.6667, i === 2 ? { acc: [1, Number.NaN, 2], rr: null } : { ts: T0 + i * 16.6667 - 2 }),
      );
      const m = r.build().motion;
      expect(rel.index).toBe(i);
      expect(rel.t).toBe(m.t[i]);
      expect(r.motionTime(i)).toBe(m.t[i]);
      expect(rel.ts).toBe(m.ts[i]);
      expect(rel.acc).toEqual(m.acc_x[i] === null || m.acc_y[i] === null ? null : [m.acc_x[i], m.acc_y[i], m.acc_z[i]]);
      expect(rel.accg).toEqual([m.accg_x[i], m.accg_y[i], m.accg_z[i]]);
      expect(rel.rr).toEqual(i === 2 ? null : { alpha: m.rr_alpha[i], beta: m.rr_beta[i], gamma: m.rr_gamma[i] });
    }
    expect(r.motionCount).toBe(5);
    expect(() => r.motionTime(5)).toThrow(RangeError);
    expect(() => r.motionTime(-1)).toThrow(RangeError);
  });

  it('orientation / quaternion streams are null when nothing was recorded; build() returns copies', () => {
    const r = new StreamRecorder(0);
    expect(r.lastTime).toBeNull();
    const empty = r.build();
    expect(empty.orientation).toBeNull();
    expect(empty.quaternion).toBeNull();
    expect(empty.motion.t).toEqual([]);
    expect(empty.motion.interval_raw).toBeNull();
    expect(streamErrors('MotionStream', empty.motion)).toEqual([]);

    r.addMotion(motion(1));
    r.addTouch(2.556, 'end', 2.4, 10.123, 20.987);
    r.addCue(3, 'ready');
    r.addOrientation({ t: 4, ts: 4, alpha: 1, beta: 2, gamma: 3, absolute: null, compass: null });
    r.addOrientation({ t: 5, ts: 5, alpha: 1, beta: 2, gamma: 3, absolute: false, compass: null });
    const a = r.build();
    expect(a.touches).toEqual([{ t: 2.56, type: 'end', n: 2, x: 10.12, y: 20.99 }]);
    expect(a.orientation?.absolute).toBe(false); // first non-null value
    a.motion.t.push(99);
    a.touches[0].t = -1;
    a.cues.push({ t: 0, kind: 'go' });
    const b = r.build();
    expect(b.motion.t).toEqual([1]);
    expect(b.touches[0].t).toBe(2.56);
    expect(b.cues).toEqual([{ t: 3, kind: 'ready' }]);
    expect(r.lastTime).toBe(5);
    expect(r.rel(10.004)).toBe(10);
    expect(r.rel(Number.NaN)).toBeNull();
  });
});

describe('rateStats', () => {
  it('returns nulls for fewer than two times and hz = null for a zero span', () => {
    expect(rateStats([])).toEqual({ n: 0, hz: null, dt_median: null, dt_p95: null, dt_max: null, gaps_over_50ms: 0 });
    expect(rateStats([5])).toEqual({ n: 1, hz: null, dt_median: null, dt_p95: null, dt_max: null, gaps_over_50ms: 0 });
    expect(rateStats([5, 5])).toMatchObject({ n: 2, hz: null, dt_median: 0, dt_max: 0 });
  });

  it('hz = (n−1)/span·1000 on a regular 60 Hz grid', () => {
    const t = Array.from({ length: 121 }, (_, i) => (i * 1000) / 60);
    const s = rateStats(t);
    expect(s.n).toBe(121);
    expect(s.hz).toBeCloseTo(60, 9);
    expect(s.dt_median).toBeCloseTo(16.6667, 4);
    expect(s.dt_p95).toBeCloseTo(16.6667, 4);
    expect(s.gaps_over_50ms).toBe(0);
  });

  it('median / p95 (linear interpolation) / max dt and gaps strictly over 50 ms', () => {
    // dts: 10, 20, 30, 50, 51, 100 → sorted; median = (30 + 50)/2 = 40; p95 at rank 4.75 → 51 + 0.75·49
    const t = [0, 10, 30, 60, 110, 161, 261];
    const s = rateStats(t);
    expect(s.n).toBe(7);
    expect(s.hz).toBeCloseTo((6 / 261) * 1000, 12);
    expect(s.dt_median).toBe(40);
    expect(s.dt_p95).toBeCloseTo(51 + 0.75 * 49, 12);
    expect(s.dt_max).toBe(100);
    expect(s.gaps_over_50ms).toBe(2); // 51 and 100; exactly 50 is not a gap
    expect(rateStats([0, 20, 10, 30]).dt_max).toBe(20); // unsorted input is measured as given
    expect(rateStats([0, Number.NaN, 10]).n).toBe(2);
  });
});

describe('computeQuality', () => {
  it('counts samples, reports availability, rates and passes warnings through (deduplicated)', () => {
    const r = new StreamRecorder(0);
    for (let i = 0; i < 61; i++) {
      r.addMotion(motion(i * 16.5, i < 3 ? { acc: null, rr: null } : {}));
      r.addOrientation({ t: i * 16.5 + 1, ts: null, alpha: 1, beta: 2, gamma: 3, absolute: false, compass: null });
    }
    r.addMotion(motion(61 * 16.5 + 80)); // one 96.5 ms gap
    const q = computeQuality(r.build(), true, ['x', 'x', 'y']);
    expect(q).toEqual({
      n_motion: 62,
      n_orientation: 61,
      n_quaternion: 0,
      motion_hz: roundValue((61 / (61 * 16.5 + 80)) * 1000, 3),
      motion_dt_ms_median: 16.5,
      motion_dt_ms_max: 96.5,
      gaps_over_50ms: 1,
      acc_available: true,
      rotation_available: true,
      hidden_during: true,
      warnings: ['x', 'y'],
    });
  });

  it('adds warnings for missing sensors, a low rate and an empty stream', () => {
    const r = new StreamRecorder(0);
    for (let i = 0; i < 20; i++) r.addMotion(motion(i * 50, { acc: null, rr: { alpha: 1, beta: Number.NaN, gamma: 1 } }));
    const q = computeQuality(r.build(), false, []);
    expect(q.acc_available).toBe(false);
    expect(q.rotation_available).toBe(false);
    expect(q.motion_hz).toBe(20);
    expect(q.motion_hz as number).toBeLessThan(LOW_MOTION_HZ);
    expect(q.warnings).toEqual(['acc_unavailable', 'rotation_unavailable', 'low_motion_rate']);
    const empty = computeQuality(new StreamRecorder(0).build(), false, ['source_ended']);
    expect(empty).toMatchObject({ n_motion: 0, motion_hz: null, motion_dt_ms_median: null, motion_dt_ms_max: null });
    expect(empty.warnings).toEqual(['source_ended', 'no_motion_samples']);
  });

  it('flags possible gyroscope and accelerometer saturation once each', () => {
    const calm = new StreamRecorder(0);
    for (let i = 0; i < 30; i++) calm.addMotion(motion(i * 16.7));
    expect(computeQuality(calm.build(), false, []).warnings).toEqual([]);

    const r = new StreamRecorder(0);
    for (let i = 0; i < 30; i++) {
      const fast = i === 10 || i === 11;
      r.addMotion(
        motion(i * 16.7, {
          rr: fast ? { alpha: 0, beta: CLIP_GYRO_DPS + 1, gamma: 0 } : { alpha: 1, beta: 1, gamma: 1 },
          accg: i === 20 ? [0, 0, CLIP_ACC_MS2 + 0.5] : [0, 9.8, 0],
        }),
      );
    }
    expect(computeQuality(r.build(), false, []).warnings).toEqual(['possible_gyro_clipping', 'possible_acc_clipping']);
  });
});
