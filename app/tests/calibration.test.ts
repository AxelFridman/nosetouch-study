import { describe, expect, it } from 'vitest';
import { axisVerdict, flatVerdict, uprightVerdict } from '../src/app/calibrationVerdicts';
import {
  axisCheck,
  buildCalibration,
  CALIBRATION_LIMITS,
  detectionCalibration,
  engineGravitySign,
  FLAT_SIGN_CONTRADICTS_ENGINE,
  flatSignContradictsEngine,
  gravityFromFlat,
  implicitCalibration,
  poseStats,
  preGoWindow,
  UPRIGHT_SIGN_CONTRADICTS_ENGINE,
  uprightCheck,
  withUprightHold,
} from '../src/core/calibration';
import type {
  AxisCheck,
  CalibrationResult,
  MotionSample,
  OrientationId,
  PlannedTrial,
  PoseStats,
  SessionRecord,
  TrialRecord,
  TrialStatus,
  Vec3,
} from '../src/core/types';
import { expectValid } from './helpers/schema';
import { makeTrialRecord } from './helpers/sessionFakes';
import { cross, encodeRotationRate, G, normalize, rotate, stillWindow, tiltGesture } from './helpers/sensorsCalib';
import { expectValidDef } from './helpers/sensorsSchema';

const DEG = Math.PI / 180;

function sample(t: number, accg: Vec3 | null, extra: Partial<MotionSample> = {}): MotionSample {
  return { t, ts: t, acc: null, accg, rr: null, interval: null, ...extra };
}

function expectVecClose(actual: Vec3 | null, expected: Vec3, digits = 6): void {
  expect(actual).not.toBeNull();
  for (let i = 0; i < 3; i++) expect(actual![i]).toBeCloseTo(expected[i], digits);
}

// ───────────────────────────── poseStats ─────────────────────────────

describe('poseStats', () => {
  it('computes the mean and the POPULATION sd of every stream', () => {
    const samples: MotionSample[] = [
      sample(0, [1, 2, 3], { acc: [0, 0, 1], rr: { alpha: 1, beta: 10, gamma: -2 } }),
      sample(16, [3, 2, 1], { acc: [0, 0, 3], rr: { alpha: 3, beta: 10, gamma: 2 } }),
    ];
    const s = poseStats(samples);
    expect(s.n).toBe(2);
    expect(s.duration_ms).toBe(16);
    expect(s.accg_mean).toEqual([2, 2, 2]);
    expect(s.accg_sd).toEqual([1, 0, 1]); // population sd: sqrt(((1-2)² + (3-2)²) / 2) = 1
    expect(s.acc_mean).toEqual([0, 0, 2]);
    expect(s.rr_mean).toEqual({ alpha: 2, beta: 10, gamma: 0 });
    expect(s.rr_sd).toEqual({ alpha: 1, beta: 0, gamma: 2 });
    expect(s.still).toBe(false); // n < 20
  });

  it('a phone at rest on a table is still and its window validates against the schema', () => {
    const s = poseStats(stillWindow({ up: [0, 0, 1], n: 60 }));
    expect(s.still).toBe(true);
    expect(s.n).toBe(60);
    expect(s.duration_ms).toBeCloseTo((59 * 1000) / 60, 6);
    expect(s.accg_mean[2]).toBeCloseTo(G, 1);
    expect(s.rr_mean!.alpha).toBeCloseTo(0.4, 0);
    expectValidDef('PoseStats', s);
  });

  it('needs at least 20 samples with a finite accg (n counts only those)', () => {
    const base = stillWindow({ up: [0, 0, 1], n: 25 });
    expect(poseStats(base.slice(0, 20)).still).toBe(true);
    expect(poseStats(base.slice(0, 19)).still).toBe(false);
    const holes = base.map((s, i) => (i % 5 === 0 ? { ...s, accg: null } : s)); // 5 dropouts
    const st = poseStats(holes);
    expect(st.n).toBe(20);
    expect(st.still).toBe(true);
    const nanned = base.map((s, i) => (i < 6 ? { ...s, accg: [NaN, 0, 9.8] as Vec3 } : s));
    expect(poseStats(nanned).n).toBe(19);
    expect(poseStats(nanned).still).toBe(false);
  });

  it('accg sd ≥ 0.25 m/s² on any axis is not still (exact boundary)', () => {
    // alternating ±a on one axis has a population sd of exactly a
    const alternating = (axis: number, a: number): MotionSample[] =>
      Array.from({ length: 40 }, (_, i) => {
        const v: Vec3 = [0, 0, 9.8];
        v[axis] += i % 2 === 0 ? a : -a;
        return sample(i * 16, v, { rr: { alpha: 0, beta: 0, gamma: 0 } });
      });
    for (const axis of [0, 1, 2]) {
      expect(poseStats(alternating(axis, 0.24)).still).toBe(true);
      expect(poseStats(alternating(axis, 0.26)).still).toBe(false);
      expect(poseStats(alternating(axis, 0.26)).accg_sd[axis]).toBeCloseTo(0.26, 12);
    }
    expect(poseStats(stillWindow({ up: [0, 0, 1], shake: 0.5 })).still).toBe(false); // sd ≈ 0.35
    expect(poseStats(stillWindow({ up: [0, 0, 1], shake: 0.1 })).still).toBe(true); // sd ≈ 0.07
  });

  it('rotation-rate sd ≥ 3 deg/s is not still; a missing gyroscope does not block stillness', () => {
    expect(poseStats(stillWindow({ up: [0, 0, 1], rrNoise: 4 })).still).toBe(false);
    const noGyro = poseStats(stillWindow({ up: [0, 0, 1], withRr: false, withAcc: false }));
    expect(noGyro.still).toBe(true);
    expect(noGyro.rr_mean).toBeNull();
    expect(noGyro.rr_sd).toBeNull();
    expect(noGyro.acc_mean).toBeNull();
    expectValidDef('PoseStats', noGyro);
  });

  it('an empty window or one without accelerometer values is never still (and stays JSON-safe)', () => {
    const empty = poseStats([]);
    expect(empty).toEqual({
      n: 0,
      duration_ms: 0,
      accg_mean: [0, 0, 0],
      accg_sd: [0, 0, 0],
      acc_mean: null,
      rr_mean: null,
      rr_sd: null,
      still: false,
    });
    expectValidDef('PoseStats', empty);
    const noAccg = poseStats(stillWindow({ up: [0, 0, 1] }).map((s) => ({ ...s, accg: null })));
    expect(noAccg.n).toBe(0);
    expect(noAccg.still).toBe(false);
  });
});

// ───────────────────────────── gravityFromFlat ─────────────────────────────

describe('gravityFromFlat', () => {
  it('flat, screen up, W3C/Android convention → sign +1', () => {
    const r = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], accgSign: 1 })));
    expect(r.sign).toBe(1);
    expect(r.magnitude!).toBeCloseTo(G, 1);
    expect(r.notes).toEqual([]);
  });

  it('flat, screen up, iOS WebKit convention (z ≈ −9.81) → sign −1', () => {
    const r = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], accgSign: -1 })));
    expect(r.sign).toBe(-1);
    expect(r.magnitude!).toBeCloseTo(G, 1);
    expect(r.notes).toEqual([]);
  });

  it('accepts a slightly tilted table (20°) but not a tilted phone (40°): "not flat"', () => {
    const tilted = (deg: number): Vec3 => rotate([0, 0, 1], [1, 0, 0], deg * DEG);
    expect(gravityFromFlat(poseStats(stillWindow({ up: tilted(20) }))).sign).toBe(1);
    const r = gravityFromFlat(poseStats(stillWindow({ up: tilted(40) })));
    expect(r.sign).toBeNull();
    expect(r.magnitude).toBeNull();
    expect(r.notes.join(' ')).toContain('not flat');
  });

  it('a moving phone is rejected with "moving"', () => {
    const r = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], shake: 1.2 })));
    expect(r).toMatchObject({ sign: null, magnitude: null });
    expect(r.notes.join(' ')).toContain('moving');
  });

  it('an implausible magnitude is rejected (e.g. a sensor reporting in g instead of m/s²)', () => {
    const r = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], magnitude: 1, accgNoise: 0.003 })));
    expect(r).toMatchObject({ sign: null, magnitude: null });
    expect(r.notes.join(' ')).toContain('implausible magnitude');
    const high = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], magnitude: 11.5 })));
    expect(high.notes.join(' ')).toContain('implausible magnitude');
    const edge = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], magnitude: 10.9, accgNoise: 0.001 })));
    expect(edge.sign).toBe(1);
  });

  it('too few samples and missing data are explained, never guessed', () => {
    const few = gravityFromFlat(poseStats(stillWindow({ up: [0, 0, 1], n: 10 })));
    expect(few.sign).toBeNull();
    expect(few.notes.join(' ')).toContain('too few samples');
    expect(gravityFromFlat(null)).toEqual({ sign: null, magnitude: null, notes: ['flat: not recorded'] });
    expect(gravityFromFlat(poseStats([])).sign).toBeNull();
  });
});

// ───────────────────────────── uprightCheck ─────────────────────────────

describe('uprightCheck', () => {
  const upright = (accgSign: 1 | -1, up: Vec3 = [0, 1, 0]): PoseStats => poseStats(stillWindow({ up, accgSign }));

  it('W3C phone upright facing the participant → ok', () => {
    expect(uprightCheck(upright(1), 1)).toBe('ok');
  });

  it('iOS phone upright (y ≈ −9.81) with the measured sign −1 → ok', () => {
    expect(uprightCheck(upright(-1), -1)).toBe('ok');
  });

  it('opposite readings → inverted (sign mismatch or phone held upside down)', () => {
    expect(uprightCheck(upright(-1), 1)).toBe('inverted');
    expect(uprightCheck(upright(1, [0, -1, 0]), 1)).toBe('inverted');
  });

  it('unknown sign is treated as +1', () => {
    expect(uprightCheck(upright(1), null)).toBe('ok');
    expect(uprightCheck(upright(-1), null)).toBe('inverted');
  });

  it('a phone that is not upright (flat, or tilted 60°) → unclear; no stats → null', () => {
    expect(uprightCheck(upright(1, [0, 0, 1]), 1)).toBe('unclear');
    expect(uprightCheck(upright(1, rotate([0, 1, 0], [1, 0, 0], 60 * DEG)), 1)).toBe('unclear'); // y = 4.9
    expect(uprightCheck(upright(1, rotate([0, 1, 0], [1, 0, 0], 40 * DEG)), 1)).toBe('ok'); // y = 7.5
    expect(uprightCheck(null, 1)).toBeNull();
  });
});

// ───────────────────────────── axisCheck ─────────────────────────────

describe('tilt-gesture generator (sanity of the test fixture itself)', () => {
  it('a positive rotation about device +y lowers the right edge: up gets a −x component', () => {
    // flat phone turned +45° about +y: device x (right edge) dips → "up" in device coords leans to −x
    const s = tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, accgNoise: 0, rrNoise: 0, rrBias: [0, 0, 0], holdMs: 600 });
    const mid = s[Math.floor(s.length / 2)];
    expect(mid.accg![0]).toBeCloseTo(-G * Math.sin(45 * DEG), 6);
    expect(mid.accg![2]).toBeCloseTo(G * Math.cos(45 * DEG), 6);
    // the gyro integral of beta (W3C: about y) over the ramp equals the tilt
    let integral = 0;
    for (let i = 1; i < s.length / 2; i++) integral += ((s[i - 1].rr!.beta + s[i].rr!.beta) / 2) * ((s[i].t - s[i - 1].t) / 1000);
    expect(integral).toBeCloseTo(45, 0);
  });

  it('encodes rotationRate per mapping', () => {
    expect(encodeRotationRate([1, 2, 3], 'w3c', 1, 'deg')).toEqual({ alpha: 1, beta: 2, gamma: 3 });
    expect(encodeRotationRate([1, 2, 3], 'legacy', 1, 'deg')).toEqual({ alpha: 3, beta: 1, gamma: 2 });
    expect(encodeRotationRate([1, 2, 3], 'w3c', -1, 'deg')).toEqual({ alpha: -1, beta: -2, gamma: -3 });
  });
});

describe('axisCheck', () => {
  // Oblique case: the phone starts tilted (up = (0, 0.5, 1)) and turns about an axis perpendicular
  // to gravity with all three device components non-zero, so w3c and legacy predictions differ.
  const obliqueUp = normalize([0, 0.5, 1]);
  const obliqueAxis = normalize(cross(obliqueUp, [1, 0.2, 0]));

  const cases: { name: string; axis: Vec3; up0?: Vec3 }[] = [
    { name: 'tilt about device +y (sideways tilt of a flat phone)', axis: [0, 1, 0] },
    { name: 'tilt about device +x (forward tilt of a flat phone)', axis: [1, 0, 0] },
    { name: 'tilt about device −y', axis: [0, -1, 0] },
    { name: 'tilt about an oblique device axis', axis: obliqueAxis, up0: obliqueUp },
  ];

  for (const c of cases) {
    describe(c.name, () => {
      const run = (over: Partial<Parameters<typeof tiltGesture>[0]> = {}): AxisCheck =>
        axisCheck(tiltGesture({ axis: c.axis, up0: c.up0, tiltDeg: 45, ...over }));

      it('W3C field mapping (alpha=x, beta=y, gamma=z) → w3c, deg/s', () => {
        const r = run({ rrEncoding: 'w3c' });
        expect(r.verdict).toBe('w3c');
        expect(r.units).toBe('deg_per_s');
        expect(r.similarity!).toBeGreaterThan(0.97);
        expect(r.magnitude_ratio!).toBeGreaterThan(0.85);
        expect(r.magnitude_ratio!).toBeLessThan(1.15);
        expect(r.tilt_deg!).toBeGreaterThan(40);
        expect(r.tilt_deg!).toBeLessThan(47);
        // the observed axis is the true device axis
        expect(r.device_axis![0] * c.axis[0] + r.device_axis![1] * c.axis[1] + r.device_axis![2] * c.axis[2]).toBeGreaterThan(0.99);
        expectValidDef('AxisCheck', r);
      });

      it('legacy mapping (alpha=z, beta=x, gamma=y) → legacy', () => {
        expect(run({ rrEncoding: 'legacy' }).verdict).toBe('legacy');
      });

      it('negated rotationRate → *_inverted', () => {
        expect(run({ rrEncoding: 'w3c', rrSign: -1 }).verdict).toBe('w3c_inverted');
        expect(run({ rrEncoding: 'legacy', rrSign: -1 }).verdict).toBe('legacy_inverted');
      });

      it('rad/s instead of deg/s → units rad_per_s (mapping still identified)', () => {
        const r = run({ rrEncoding: 'w3c', rrUnits: 'rad' });
        expect(r.units).toBe('rad_per_s');
        expect(r.verdict).toBe('w3c');
        expect(r.magnitude_ratio!).toBeCloseTo(1 / 57.2958, 2);
      });

      it('iOS accelerometer sign (accg negated) gives the SAME result', () => {
        // exactly the same window with every accg vector negated (noise included)
        const window = tiltGesture({ axis: c.axis, up0: c.up0, tiltDeg: 45, rrEncoding: 'w3c' });
        const negated = window.map((s) => ({ ...s, accg: s.accg && ([-s.accg[0], -s.accg[1], -s.accg[2]] as Vec3) }));
        const w3cSign = axisCheck(window);
        const iosSign = axisCheck(negated);
        expect(iosSign.verdict).toBe('w3c');
        expect(iosSign.verdict).toBe(w3cSign.verdict);
        expect(iosSign.tilt_deg!).toBeCloseTo(w3cSign.tilt_deg!, 9);
        expect(iosSign.similarity!).toBeCloseTo(w3cSign.similarity!, 9);
        expect(iosSign.magnitude_ratio!).toBeCloseTo(w3cSign.magnitude_ratio!, 9);
        expectVecClose(iosSign.device_axis, w3cSign.device_axis!, 9);
        // and a window generated with the iOS sign (independent noise) agrees on every verdict
        expect(run({ rrEncoding: 'w3c', accgSign: -1 }).verdict).toBe('w3c');
        expect(run({ rrEncoding: 'legacy', rrSign: -1, accgSign: -1 }).verdict).toBe('legacy_inverted');
      });
    });
  }

  it('a tilt smaller than 30° is unclear (diagnostics still reported, units withheld)', () => {
    const r = axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 20 }));
    expect(r.verdict).toBe('unclear');
    expect(r.units).toBeNull();
    expect(r.tilt_deg!).toBeGreaterThan(17);
    expect(r.tilt_deg!).toBeLessThan(CALIBRATION_LIMITS.axis_min_tilt_deg);
    expect(r.rr_integral_deg).not.toBeNull();
    expectValidDef('AxisCheck', r);
    const tiny = axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 0 }));
    expect(tiny.verdict).toBe('unclear');
    expect(tiny.device_axis).toBeNull(); // a sub-degree "axis" would be noise
    expect(tiny.similarity).toBeNull();
    expectValidDef('AxisCheck', tiny);
  });

  it('works at 100 Hz (Firefox) and with jittery timing', () => {
    expect(axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 50, hz: 100 })).verdict).toBe('w3c');
    const jittered = tiltGesture({ axis: [1, 0, 0], tiltDeg: 50, rrEncoding: 'legacy' }).map((s, i) => ({
      ...s,
      t: s.t + (i % 3) * 4, // ±4 ms arrival jitter (handler time), still increasing
    }));
    expect(axisCheck(jittered).verdict).toBe('legacy');
  });

  it('a gyro-less device (rotationRate null) is unclear, but the tilt is still measured', () => {
    const r = axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, noRr: true }));
    expect(r.verdict).toBe('unclear');
    expect(r.rr_integral_deg).toBeNull();
    expect(r.units).toBeNull();
    expect(r.tilt_deg!).toBeGreaterThan(40);
    expectValidDef('AxisCheck', r);
  });

  it('rotation that disagrees with gravity (similarity < 0.85) is unclear but units are kept', () => {
    // Gravity says "about +y"; the gyro puts the same amount of rotation into alpha and gamma
    // (w3c reads that as x+z, legacy as y+z): no mapping explains it — best cosine is 1/√2.
    const s = tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, rrEncoding: 'w3c' }).map((x) => ({
      ...x,
      rr: x.rr && { alpha: x.rr.beta / Math.SQRT2, beta: 0, gamma: x.rr.beta / Math.SQRT2 },
    }));
    const r = axisCheck(s);
    expect(r.verdict).toBe('unclear');
    expect(r.similarity!).toBeLessThan(0.85);
    expect(r.similarity!).toBeGreaterThan(0.6);
    expect(r.units).toBe('deg_per_s');
  });

  it('a magnitude far from both unit ranges → unclear with units null', () => {
    const s = tiltGesture({ axis: [0, 1, 0], tiltDeg: 45 }).map((x) => ({
      ...x,
      rr: x.rr && { alpha: x.rr.alpha * 5, beta: x.rr.beta * 5, gamma: x.rr.gamma * 5 },
    }));
    const r = axisCheck(s);
    expect(r.verdict).toBe('unclear');
    expect(r.units).toBeNull();
    expect(r.magnitude_ratio!).toBeGreaterThan(4);
  });

  it('empty and degenerate windows', () => {
    const empty = axisCheck([]);
    expect(empty).toMatchObject({ n: 0, duration_ms: 0, tilt_deg: null, verdict: 'unclear', units: null });
    expectValidDef('AxisCheck', empty);
    const one = axisCheck([sample(0, [0, 0, 9.8], { rr: { alpha: 0, beta: 0, gamma: 0 } })]);
    expect(one.verdict).toBe('unclear');
    expectValidDef('AxisCheck', one);
  });
});

// ───────────────────────────── buildCalibration ─────────────────────────────

describe('buildCalibration', () => {
  it('assembles a complete Android-style calibration that validates against the schema', () => {
    const flat = poseStats(stillWindow({ up: [0, 0, 1], accgSign: 1 }));
    const upright = poseStats(stillWindow({ up: [0, 1, 0], accgSign: 1 }));
    const axis = axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45 }));
    const cal = buildCalibration({ performedAt: '2026-10-23T10:00:00.000Z', flat, upright, axis });
    expect(cal.gravity_sign).toBe(1);
    expect(cal.gravity_magnitude!).toBeCloseTo(G, 1);
    expect(cal.upright_check).toBe('ok');
    expect(cal.gyro_bias_dps).toEqual(flat.rr_mean);
    expect(cal.axis_check?.verdict).toBe('w3c');
    expect(cal.notes).toEqual([]);
    expect(cal.flat).toBe(flat);
    expect(cal.upright).toBe(upright);
    expect(cal.method).toBe('guided_v1'); // a measured (screen) calibration says so; v2 sessions are implicit_v1
    expectValid('CalibrationResult', cal);
  });

  it('iOS-style calibration: sign −1 and upright ok', () => {
    const cal = buildCalibration({
      performedAt: '2026-10-23T10:00:00.000Z',
      flat: poseStats(stillWindow({ up: [0, 0, 1], accgSign: -1 })),
      upright: poseStats(stillWindow({ up: [0, 1, 0], accgSign: -1 })),
      axis: axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, accgSign: -1 })),
      notes: ['participant note'],
    });
    expect(cal.gravity_sign).toBe(-1);
    expect(cal.upright_check).toBe('ok');
    expect(cal.axis_check?.verdict).toBe('w3c');
    expect(cal.notes).toEqual(['participant note']);
    expectValid('CalibrationResult', cal);
  });

  it('problems end up in notes; a skipped calibration is still a valid record', () => {
    const moving = buildCalibration({
      performedAt: '2026-10-23T10:00:00.000Z',
      flat: poseStats(stillWindow({ up: [0, 0, 1], shake: 1.5 })),
      upright: poseStats(stillWindow({ up: [0, -1, 0] })),
      axis: axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, rrEncoding: 'legacy', rrUnits: 'rad' })),
    });
    expect(moving.gravity_sign).toBeNull();
    expect(moving.upright_check).toBe('inverted');
    expect(moving.gyro_bias_dps).toBeNull(); // a moving window measured the hand, not the gyro bias
    const text = moving.notes.join('\n');
    expect(text).toContain('moving');
    expect(text).toContain('upright: inverted');
    expect(text).toContain('axis check: legacy');
    expectValid('CalibrationResult', moving);

    const skipped = buildCalibration({ performedAt: '2026-10-23T10:00:00.000Z', flat: null, upright: null, axis: null });
    expect(skipped).toMatchObject({
      flat: null,
      upright: null,
      gravity_sign: null,
      gravity_magnitude: null,
      upright_check: null,
      gyro_bias_dps: null,
      axis_check: null,
    });
    expectValid('CalibrationResult', skipped);
  });

  it('notes rad/s even when the mapping is w3c', () => {
    const cal = buildCalibration({
      performedAt: '2026-10-23T10:00:00.000Z',
      flat: poseStats(stillWindow({ up: [0, 0, 1] })),
      upright: null,
      axis: axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, rrUnits: 'rad' })),
    });
    expect(cal.axis_check?.verdict).toBe('w3c');
    expect(cal.notes.join(' ')).toContain('rad/s');
    expectValid('CalibrationResult', cal);
  });

  it('stores the rest rate as gyro bias only from a still flat window that gave a gravity sign', () => {
    const at = '2026-10-23T10:00:00.000Z';
    const bias = { alpha: 5, beta: -3, gamma: 2 }; // deg/s, far above a real bias: easy to see
    const build = (flat: PoseStats | null) => buildCalibration({ performedAt: at, flat, upright: null, axis: null });
    // A hand left on the phone: accg barely moves, the gyroscope sees the tremor → "moving", no bias.
    const handOn = poseStats(stillWindow({ up: [0, 0, 1], rrBias: bias, rrNoise: 4 }));
    expect(handOn.still).toBe(false);
    expect(handOn.rr_mean!.alpha).toBeCloseTo(5, 0);
    const moving = build(handOn);
    expect(moving.gravity_sign).toBeNull();
    expect(moving.notes.join(' ')).toContain('moving');
    expect(moving.gyro_bias_dps).toBeNull();
    expectValid('CalibrationResult', moving);
    // Still, but rejected for another reason (tilted 40°, implausible magnitude): no bias either.
    const tilted = poseStats(stillWindow({ up: rotate([0, 0, 1], [1, 0, 0], 40 * DEG), rrBias: bias }));
    expect(tilted.still).toBe(true);
    expect(build(tilted).gravity_sign).toBeNull();
    expect(build(tilted).gyro_bias_dps).toBeNull();
    const odd = poseStats(stillWindow({ up: [0, 0, 1], magnitude: 1, accgNoise: 0.003, rrBias: bias }));
    expect(odd.still).toBe(true);
    expect(build(odd).gyro_bias_dps).toBeNull();
    // A good window: the mean rest rate. Without a gyroscope there is nothing to store.
    const good = poseStats(stillWindow({ up: [0, 0, 1], rrBias: bias }));
    expect(build(good).gyro_bias_dps).toEqual(good.rr_mean);
    expect(build(good).gyro_bias_dps!.beta).toBeCloseTo(-3, 0);
    expect(build(poseStats(stillWindow({ up: [0, 0, 1], withRr: false }))).gyro_bias_dps).toBeNull();
  });
});

// ───────────────────────────── engine expectation / screen-down flat step ─────────────────────────────

describe('flat sign against the engine expectation', () => {
  const at = '2026-10-23T10:00:00.000Z';
  // Lying screen-DOWN still gives |z| ≈ |g|: the sign flips but the window passes every check.
  const androidUp = poseStats(stillWindow({ up: [0, 0, 1], accgSign: 1 }));
  const androidDown = poseStats(stillWindow({ up: [0, 0, -1], accgSign: 1 }));
  const iosUp = poseStats(stillWindow({ up: [0, 0, 1], accgSign: -1 }));
  const iosDown = poseStats(stillWindow({ up: [0, 0, -1], accgSign: -1 }));
  const androidUpright = poseStats(stillWindow({ up: [0, 1, 0], accgSign: 1 }));
  const iosUpright = poseStats(stillWindow({ up: [0, 1, 0], accgSign: -1 }));
  const sign = (p: PoseStats) => gravityFromFlat(p).sign;

  it('engine expectation: webkit −1, blink / gecko +1, unknown or absent → none', () => {
    expect(engineGravitySign('webkit')).toBe(-1);
    expect(engineGravitySign('blink')).toBe(1);
    expect(engineGravitySign('gecko')).toBe(1);
    expect(engineGravitySign('unknown')).toBeNull();
    expect(engineGravitySign(undefined)).toBeNull();
    expect(engineGravitySign(null)).toBeNull();
    expect(flatSignContradictsEngine(-1, 'blink')).toBe(true);
    expect(flatSignContradictsEngine(1, 'webkit')).toBe(true);
    expect(flatSignContradictsEngine(-1, 'webkit')).toBe(false);
    expect(flatSignContradictsEngine(1, 'gecko')).toBe(false);
    expect(flatSignContradictsEngine(null, 'blink')).toBe(false);
    expect(flatSignContradictsEngine(-1, 'unknown')).toBe(false);
    expect(flatSignContradictsEngine(1, undefined)).toBe(false);
  });

  it('a screen-down window passes the flat checks with the wrong sign (why the engine check exists)', () => {
    expect(sign(androidDown)).toBe(-1);
    expect(sign(iosDown)).toBe(1);
    expect(gravityFromFlat(androidDown).notes).toEqual([]);
  });

  it('screen-down on Android: the sign is kept but noted, and the upright step then reads inverted', () => {
    const cal = buildCalibration({ performedAt: at, flat: androidDown, upright: androidUpright, axis: null, engine: 'blink' });
    expect(cal.gravity_sign).toBe(-1); // kept: the analysis weighs it against the trial votes
    expect(cal.upright_check).toBe('inverted');
    expect(cal.notes).toContain(FLAT_SIGN_CONTRADICTS_ENGINE);
    expect(cal.notes.join('\n')).toContain('upright: inverted');
    expect(cal.gyro_bias_dps).toEqual(androidDown.rr_mean); // still and signed: the rest rate is valid
    expectValid('CalibrationResult', cal);
  });

  it('screen-down on an iPhone (WebKit) is noted the same way', () => {
    const cal = buildCalibration({ performedAt: at, flat: iosDown, upright: iosUpright, axis: null, engine: 'webkit' });
    expect(cal.gravity_sign).toBe(1);
    expect(cal.upright_check).toBe('inverted');
    expect(cal.notes).toContain(FLAT_SIGN_CONTRADICTS_ENGINE);
    expectValid('CalibrationResult', cal);
  });

  it('no note when the sign matches the engine, or when there is no expectation (unknown engine, simulation)', () => {
    const notesOf = (flat: PoseStats, engine?: 'webkit' | 'blink' | 'gecko' | 'unknown' | null) =>
      buildCalibration({ performedAt: at, flat, upright: null, axis: null, engine }).notes;
    expect(notesOf(androidUp, 'blink')).toEqual([]);
    expect(notesOf(androidUp, 'gecko')).toEqual([]);
    expect(notesOf(iosUp, 'webkit')).toEqual([]);
    expect(notesOf(androidDown, 'unknown')).toEqual([]);
    expect(notesOf(androidDown, null)).toEqual([]);
    expect(notesOf(androidDown)).toEqual([]);
  });

  it('the verdict judges the flat result against the engine: a contradicted sign is "face_down"', () => {
    expect(flatVerdict(androidUp, sign(androidUp), engineGravitySign('blink'))).toBe('ok');
    expect(flatVerdict(androidDown, sign(androidDown), engineGravitySign('blink'))).toBe('face_down');
    expect(flatVerdict(iosUp, sign(iosUp), engineGravitySign('webkit'))).toBe('ok_inverted');
    expect(flatVerdict(iosDown, sign(iosDown), engineGravitySign('webkit'))).toBe('face_down');
    // no expectation: either sign is accepted
    expect(flatVerdict(androidDown, sign(androidDown), engineGravitySign('unknown'))).toBe('ok_inverted');
    expect(flatVerdict(androidUp, sign(androidUp))).toBe('ok');
    // rejected windows keep their own reason whatever the engine
    const moving = poseStats(stillWindow({ up: [0, 0, 1], shake: 1.5 }));
    expect(flatVerdict(moving, sign(moving), 1)).toBe('moving');
    expect(flatVerdict(null, null, 1)).toBeNull();
    // and the upright step of the screen-down session is reported as inverted
    expect(uprightVerdict(androidUpright, uprightCheck(androidUpright, sign(androidDown)))).toBe('inverted');
  });

  // v2 has no calibration screen (spec §1), so its result texts are gone; the verdict rules stay.
  it('the remaining verdicts: a still flat window that is not flat, odd magnitudes, upright and axis results', () => {
    const tilted = poseStats(stillWindow({ up: rotate([0, 0, 1], [1, 0, 0], 40 * DEG) }));
    expect(flatVerdict(tilted, sign(tilted), 1)).toBe('not_flat');
    const inG = poseStats(stillWindow({ up: [0, 0, 1], magnitude: 1, accgNoise: 0.003 }));
    expect(flatVerdict(inG, sign(inG), 1)).toBe('odd');
    expect(uprightVerdict(androidUpright, uprightCheck(androidUpright, 1))).toBe('ok');
    const flatAsUpright = poseStats(stillWindow({ up: [0, 0, 1] }));
    expect(uprightVerdict(flatAsUpright, uprightCheck(flatAsUpright, 1))).toBe('unclear');
    // not upright AND not still: the hand moved (an upright reading counts whatever the wobble)
    const shaking = poseStats(stillWindow({ up: [0, 0, 1], shake: 1.5 }));
    expect(uprightVerdict(shaking, uprightCheck(shaking, 1))).toBe('moving');
    expect(uprightVerdict(null, null)).toBeNull();
    expect(axisVerdict(axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45 })))).toBe('ok');
    expect(axisVerdict(axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, rrEncoding: 'legacy' })))).toBe('noted');
    expect(axisVerdict(axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 45, rrUnits: 'rad' })))).toBe('radians');
    expect(axisVerdict(axisCheck(tiltGesture({ axis: [0, 1, 0], tiltDeg: 20 })))).toBe('unclear');
    expect(axisVerdict(null)).toBeNull();
  });
});

// ───────────────────────────── implicit calibration (implicit_v1, spec v2 §5) ─────────────────────────────

const AT = '2026-10-24T10:00:00.000Z';

/**
 * A trial whose phone rests from 0 to 1000 ms with the given accg. Its pre-GO hold is 100…900 ms
 * (makeTrialRecord: still_ms 500, go_ms 900; still_hold_ms 400 in the v2 parameters). `wobble`
 * alternates ± that much on x (population sd = wobble).
 */
function holdTrial(
  o: { accg?: Vec3; wobble?: number; orientation?: OrientationId; status?: TrialStatus; goMs?: number | null } = {},
): TrialRecord {
  const session = { session_id: 'S-1', participant_id: 'P-ABC234', study_code: 'test' } as SessionRecord;
  const planned: PlannedTrial = {
    trial_index: 0,
    block_index: 0,
    trial_in_block: 0,
    condition: {
      hand: 'right',
      orientation: o.orientation ?? 'upright_toward',
      speed: 'natural',
      start_position: 'natural',
      variant: 'standard',
      posture: 'standing',
      rep: 0,
      hand_source: 'instructed',
    },
  };
  const rec = makeTrialRecord(session, planned, 1, { status: o.status, timing: o.goMs === undefined ? {} : { go_ms: o.goMs } });
  const g = o.accg ?? [0, 9.81, 0];
  const m = rec.streams.motion;
  m.t.forEach((t, i) => {
    if (t >= 1000) return;
    m.accg_x[i] = g[0] + (o.wobble ? (i % 2 === 0 ? o.wobble : -o.wobble) : 0);
    m.accg_y[i] = g[1];
    m.accg_z[i] = g[2];
  });
  return rec;
}

describe('implicit calibration (implicit_v1)', () => {
  it('starts empty: every measured field null, no notes, no axis check; schema-valid', () => {
    const cal = implicitCalibration(AT);
    expect(cal).toEqual({
      method: 'implicit_v1',
      performed_at: AT,
      flat: null,
      upright: null,
      gravity_sign: null,
      gravity_magnitude: null,
      upright_check: null,
      gyro_bias_dps: null,
      notes: [],
      axis_check: null,
    });
    expectValid('CalibrationResult', cal);
  });

  it('preGoWindow is the stored hold before GO: still_ms − still_hold_ms … go_ms', () => {
    const rec = holdTrial();
    const w = preGoWindow(rec);
    const from = rec.timing.still_ms! - rec.segmentation.params.still_hold_ms;
    expect(w.map((x) => x.t)).toEqual(rec.streams.motion.t.filter((t) => t >= from && t <= rec.timing.go_ms!));
    expect(w.length).toBeGreaterThanOrEqual(20); // enough for poseStats to call it still
    expect(w[0].accg).toEqual([0, 9.81, 0]);
    expect(preGoWindow(holdTrial({ goMs: null }))).toEqual([]);
  });

  it('fills the sign, magnitude and upright check from the first still upright hold (Android, W3C)', () => {
    const before = implicitCalibration(AT);
    const cal = withUprightHold(before, holdTrial(), 'blink')!;
    expect(cal).not.toBe(before);
    expect(before.upright).toBeNull(); // the stored object is not changed
    expect(cal).toMatchObject({ method: 'implicit_v1', performed_at: AT, gravity_sign: 1, upright_check: 'ok', notes: [] });
    expect(cal.upright?.still).toBe(true);
    expect(cal.gravity_magnitude!).toBeCloseTo(9.81, 6);
    // an in-hand hold measures tremor, never the gyroscope's bias; the axis mapping is the analysis' job
    expect(cal.gyro_bias_dps).toBeNull();
    expect(cal.axis_check).toBeNull();
    expect(cal.flat).toBeNull();
    expectValid('CalibrationResult', cal);
  });

  it('iPhone (WebKit) reports upright as y ≈ −9.81: sign −1 is ok there', () => {
    const cal = withUprightHold(implicitCalibration(AT), holdTrial({ accg: [0, -9.81, 0] }), 'webkit')!;
    expect(cal).toMatchObject({ gravity_sign: -1, upright_check: 'ok', notes: [] });
  });

  it('a sign against the engine prior is kept, marked inverted and noted; no prior accepts either sign', () => {
    const cal = withUprightHold(implicitCalibration(AT), holdTrial({ accg: [0, 9.81, 0] }), 'webkit')!;
    expect(cal).toMatchObject({ gravity_sign: 1, upright_check: 'inverted', notes: [UPRIGHT_SIGN_CONTRADICTS_ENGINE] });
    expectValid('CalibrationResult', cal);
    for (const engine of ['unknown', null, undefined] as const) {
      expect(withUprightHold(implicitCalibration(AT), holdTrial({ accg: [0, -9.81, 0] }), engine)?.upright_check).toBe('ok');
    }
  });

  it('a still hold that is not upright enough (|y| ≤ 7) is unclear: no sign; an implausible |g| is not stored', () => {
    const leaning: Vec3 = [0, 6, 7.76]; // ≈ 52° back from upright, |g| ≈ 9.81
    const cal = withUprightHold(implicitCalibration(AT), holdTrial({ accg: leaning }), 'blink')!;
    expect(cal.upright).not.toBeNull();
    expect(cal).toMatchObject({ gravity_sign: null, upright_check: 'unclear', notes: [] });
    expect(cal.gravity_magnitude!).toBeCloseTo(Math.hypot(...leaning), 6);
    const odd = withUprightHold(implicitCalibration(AT), holdTrial({ accg: [0, 12, 0] }), 'blink')!;
    expect(odd.gravity_sign).toBe(1);
    expect(odd.gravity_magnitude).toBeNull();
  });

  it('changes nothing for a moving hold (the next upright trial is tried) or any other trial', () => {
    const cal = implicitCalibration(AT);
    const moving = holdTrial({ wobble: 0.6 });
    expect(poseStats(preGoWindow(moving)).still).toBe(false);
    expect(withUprightHold(cal, moving, 'blink')).toBe(cal);
    expect(withUprightHold(cal, holdTrial({ orientation: 'top_left_toward' }), 'blink')).toBe(cal);
    expect(withUprightHold(cal, holdTrial({ status: 'no_movement' }), 'blink')).toBe(cal);
    expect(withUprightHold(cal, holdTrial({ goMs: null }), 'blink')).toBe(cal);
    expect(withUprightHold(null, holdTrial(), 'blink')).toBeNull();
    // only the FIRST still upright hold counts, and never on top of a guided calibration
    const filled = withUprightHold(cal, holdTrial(), 'blink')!;
    expect(withUprightHold(filled, holdTrial({ accg: [0, -9.81, 0] }), 'blink')).toBe(filled);
    const guided = buildCalibration({ performedAt: AT, flat: null, upright: null, axis: null });
    expect(withUprightHold(guided, holdTrial(), 'blink')).toBe(guided);
    const legacy: CalibrationResult = { ...implicitCalibration(AT), method: undefined };
    expect(withUprightHold(legacy, holdTrial(), 'blink')).toBe(legacy);
  });
});

describe('detectionCalibration (the online grip classification)', () => {
  const measured = (sign: 1 | -1 | null, notes: string[] = []): CalibrationResult => ({
    ...implicitCalibration(AT),
    gravity_sign: sign,
    notes,
  });

  it('a trusted sign passes through as the very same object', () => {
    const cal = measured(1);
    expect(detectionCalibration(cal, 'blink', AT)).toBe(cal);
    expect(detectionCalibration(cal, 'unknown', AT)).toBe(cal);
    const ios = measured(-1);
    expect(detectionCalibration(ios, 'webkit', AT)).toBe(ios);
  });

  it('while the sign is unknown the engine prior stands in, without changing the stored object', () => {
    const cal = implicitCalibration(AT);
    expect(detectionCalibration(cal, 'webkit', AT)).toEqual({ ...cal, gravity_sign: -1 });
    expect(detectionCalibration(cal, 'blink', AT)?.gravity_sign).toBe(1);
    expect(cal.gravity_sign).toBeNull();
    expect(detectionCalibration(cal, null, AT)).toBe(cal); // nothing to change: the same object
    expect(detectionCalibration(cal, 'unknown', AT)).toBe(cal);
  });

  it('a sign that contradicts the engine is distrusted, noted or not', () => {
    expect(detectionCalibration(measured(1, [UPRIGHT_SIGN_CONTRADICTS_ENGINE]), 'webkit', AT)?.gravity_sign).toBe(-1);
    expect(detectionCalibration(measured(-1, [FLAT_SIGN_CONTRADICTS_ENGINE]), 'blink', AT)?.gravity_sign).toBe(1);
    expect(detectionCalibration(measured(-1), 'gecko', AT)?.gravity_sign).toBe(1);
    // noted, but no prior to fall back to: unknown
    expect(detectionCalibration(measured(-1, [FLAT_SIGN_CONTRADICTS_ENGINE]), 'unknown', AT)?.gravity_sign).toBeNull();
  });

  it('without any calibration: a stand-in with the engine sign, or nothing', () => {
    expect(detectionCalibration(null, 'webkit', AT)).toMatchObject({ performed_at: AT, gravity_sign: -1, flat: null, notes: [] });
    expect(detectionCalibration(null, 'unknown', AT)).toBeNull();
    expect(detectionCalibration(null, null, AT)).toBeNull();
  });
});
