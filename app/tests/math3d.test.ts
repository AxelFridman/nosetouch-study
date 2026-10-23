import { describe, expect, it } from 'vitest';
import {
  DEG,
  angleBetweenDeg,
  eulerZXYFromMat,
  mAxisAngle,
  mDet,
  mExp,
  mFromColumns,
  mIdentity,
  mLog,
  mMul,
  mT,
  mVec,
  matFromEulerZXY,
  matFromQuat,
  quatFromMat,
  rotationRateToVector,
  vAdd,
  vCross,
  vDot,
  vNorm,
  vNormalize,
  vScale,
  vSub,
  vectorToRotationRate,
  type Mat3,
  type Quat,
} from '../src/core/math3d';
import { simRng } from '../src/sim/model';
import type { Vec3 } from '../src/core/types';

const rng = simRng(12345);

function randUnit(r: () => number): Vec3 {
  for (;;) {
    const v: Vec3 = [2 * r() - 1, 2 * r() - 1, 2 * r() - 1];
    const n = vNorm(v);
    if (n > 0.1 && n <= 1) return vScale(v, 1 / n);
  }
}

/** Uniformly distributed random rotation (via a random unit quaternion). */
function randRotation(r: () => number): Mat3 {
  const g = (): number => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  return matFromQuat([g(), g(), g(), g()]);
}

function expectMatClose(a: Mat3, b: Mat3, tol = 1e-9): void {
  for (let i = 0; i < 9; i++) expect(Math.abs(a[i] - b[i]), `entry ${i}: ${a} vs ${b}`).toBeLessThan(tol);
}

function expectVecClose(a: Vec3, b: Vec3, tol = 1e-12): void {
  for (let i = 0; i < 3; i++) expect(Math.abs(a[i] - b[i]), `component ${i}: ${a} vs ${b}`).toBeLessThan(tol);
}

function expectInW3cRange(e: { alpha: number; beta: number; gamma: number }): void {
  expect(e.alpha).toBeGreaterThanOrEqual(0);
  expect(e.alpha).toBeLessThan(360);
  expect(e.beta).toBeGreaterThanOrEqual(-180);
  expect(e.beta).toBeLessThan(180);
  expect(e.gamma).toBeGreaterThanOrEqual(-90);
  expect(e.gamma).toBeLessThan(90);
}

/** Smallest signed difference of two angles in degrees. */
function angDiff(a: number, b: number): number {
  return ((((a - b) % 360) + 540) % 360) - 180;
}

describe('vectors', () => {
  it('basic operations', () => {
    expect(vAdd([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
    expect(vSub([1, 2, 3], [4, 5, 6])).toEqual([-3, -3, -3]);
    expect(vScale([1, -2, 3], 2)).toEqual([2, -4, 6]);
    expect(vDot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(vCross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]); // x × y = z (right-handed)
    expect(vCross([0, 1, 0], [0, 0, 1])).toEqual([1, 0, 0]);
    expect(vNorm([3, 4, 12])).toBe(13);
    expect(vNormalize([0, 0, 0])).toEqual([0, 0, 0]);
    expectVecClose(vNormalize([0, 3, 4]), [0, 0.6, 0.8]);
  });

  it('angleBetweenDeg', () => {
    expect(angleBetweenDeg([1, 0, 0], [0, 1, 0])).toBeCloseTo(90, 12);
    expect(angleBetweenDeg([1, 0, 0], [2, 0, 0])).toBe(0);
    expect(angleBetweenDeg([1, 0, 0], [-1, 0, 0])).toBeCloseTo(180, 12);
    expect(angleBetweenDeg([1, 1, 0], [1, 0, 0])).toBeCloseTo(45, 12);
    expect(Number.isNaN(angleBetweenDeg([0, 0, 0], [1, 0, 0]))).toBe(true);
    // precise for tiny angles, where acos would return 0
    expect(angleBetweenDeg([1, 0, 0], [1, 1e-9, 0])).toBeCloseTo(1e-9 / DEG, 15);
  });
});

describe('matrices', () => {
  it('row-major layout, product, transpose, determinant', () => {
    const a: Mat3 = [1, 2, 3, 4, 5, 6, 7, 8, 10];
    expect(mVec(a, [1, 0, 0])).toEqual([1, 4, 7]); // first column
    expect(mT(a)).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 10]);
    expect(mDet(a)).toBeCloseTo(-3, 12);
    expect(mMul(mIdentity(), a)).toEqual(a);
    expect(mMul(a, mIdentity())).toEqual(a);
    expect(mMul([0, 1, 0, 0, 0, 1, 1, 0, 0], a)).toEqual([4, 5, 6, 7, 8, 10, 1, 2, 3]);
    const c = mFromColumns([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    expect(mVec(c, [0, 1, 0])).toEqual([4, 5, 6]);
    expect(c).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9]);
  });

  it('right-hand rule: +90° about z takes x to y, about x takes y to z, about y takes z to x', () => {
    expectVecClose(mVec(mAxisAngle([0, 0, 1], Math.PI / 2), [1, 0, 0]), [0, 1, 0]);
    expectVecClose(mVec(mAxisAngle([1, 0, 0], Math.PI / 2), [0, 1, 0]), [0, 0, 1]);
    expectVecClose(mVec(mAxisAngle([0, 1, 0], Math.PI / 2), [0, 0, 1]), [1, 0, 0]);
    // non-unit axes are normalized
    expectVecClose(mVec(mAxisAngle([0, 0, 5], Math.PI / 2), [1, 0, 0]), [0, 1, 0]);
  });

  it('random rotations are orthonormal with det +1', () => {
    for (let i = 0; i < 200; i++) {
      const r = mAxisAngle(randUnit(rng), (2 * rng() - 1) * 4);
      expect(mDet(r)).toBeCloseTo(1, 12);
      expectMatClose(mMul(r, mT(r)), mIdentity(), 1e-12);
    }
  });

  it('mExp and mLog are inverse (including angles near 0 and π)', () => {
    const angles = [0, 1e-14, 1e-9, 1e-6, 0.3, 1, 2, 3, Math.PI - 1e-6, Math.PI - 1e-3];
    for (let i = 0; i < 300; i++) {
      const axis = randUnit(rng);
      const angle = i < angles.length ? angles[i] : rng() * (Math.PI - 1e-3);
      const w = vScale(axis, angle);
      expectVecClose(mLog(mExp(w)), w, 1e-9);
      const r = randRotation(rng);
      expectMatClose(mExp(mLog(r)), r, 1e-9);
    }
    expect(mLog(mIdentity())).toEqual([0, 0, 0]);
    // exactly π: either sign of the axis is a valid answer
    const half = mLog(mAxisAngle([0, 0, 1], Math.PI));
    expect(Math.abs(half[2])).toBeCloseTo(Math.PI, 12);
    expectMatClose(mExp(half), mAxisAngle([0, 0, 1], Math.PI), 1e-12);
  });

  it('mExp matches the Rodrigues formula of mAxisAngle', () => {
    for (let i = 0; i < 50; i++) {
      const axis = randUnit(rng);
      const angle = rng() * 3;
      expectMatClose(mExp(vScale(axis, angle)), mAxisAngle(axis, angle), 1e-14);
    }
  });
});

describe('quaternions', () => {
  it('quat ↔ mat round trips, unit norm, canonical w ≥ 0', () => {
    const special: Mat3[] = [
      mIdentity(),
      mAxisAngle([1, 0, 0], Math.PI), // exercises every Shepperd branch
      mAxisAngle([0, 1, 0], Math.PI),
      mAxisAngle([0, 0, 1], Math.PI),
      mAxisAngle([1, 1, 0], Math.PI * 0.999),
      mAxisAngle([0, 1, 1], Math.PI * 0.95),
    ];
    for (let i = 0; i < 300; i++) {
      const r = i < special.length ? special[i] : randRotation(rng);
      const q = quatFromMat(r);
      expect(Math.hypot(...q)).toBeCloseTo(1, 12);
      expect(q[3]).toBeGreaterThanOrEqual(0);
      expectMatClose(matFromQuat(q), r, 1e-12);
    }
  });

  it('matFromQuat follows the Hamilton convention (q of +90° about z rotates x to y)', () => {
    const s = Math.SQRT1_2;
    const q: Quat = [0, 0, s, s];
    expectVecClose(mVec(matFromQuat(q), [1, 0, 0]), [0, 1, 0]);
    expect(quatFromMat(mAxisAngle([0, 0, 1], Math.PI / 2)).map((v) => +v.toFixed(12))).toEqual([0, 0, +s.toFixed(12), +s.toFixed(12)]);
    // q and −q are the same rotation; slightly non-unit input is normalized
    expectMatClose(matFromQuat([0, 0, -2 * s, -2 * s]), matFromQuat(q), 1e-12);
  });
});

describe('W3C Euler angles (Z-X′-Y″)', () => {
  it('known examples', () => {
    // alpha = 0, beta = 90, gamma = 0: phone upright, device y (top edge) points to the sky
    const upright = matFromEulerZXY(0, 90, 0);
    expectVecClose(mVec(upright, [0, 1, 0]), [0, 0, 1]);
    expectVecClose(mVec(upright, [0, 0, 1]), [0, -1, 0]); // screen faces South (towards the user)
    // alpha = 90: flat phone turned counter-clockwise, its top points West
    expectVecClose(mVec(matFromEulerZXY(90, 0, 0), [0, 1, 0]), [-1, 0, 0]);
    // positive gamma tilts the right edge down
    expect(mVec(matFromEulerZXY(0, 0, 30), [1, 0, 0])[2]).toBeCloseTo(-0.5, 12);
    // positive beta lifts the top edge
    expect(mVec(matFromEulerZXY(0, 30, 0), [0, 1, 0])[2]).toBeCloseTo(0.5, 12);
    // flat, screen up = identity
    expectMatClose(matFromEulerZXY(0, 0, 0), mIdentity(), 1e-15);
    // composition order: R = Rz(alpha)·Rx(beta)·Ry(gamma)
    const r = matFromEulerZXY(30, 40, 50);
    const composed = mMul(mMul(mAxisAngle([0, 0, 1], 30 * DEG), mAxisAngle([1, 0, 0], 40 * DEG)), mAxisAngle([0, 1, 0], 50 * DEG));
    expectMatClose(r, composed, 1e-14);
    expect(mDet(r)).toBeCloseTo(1, 12);
  });

  it('euler → mat → euler round trip (away from gimbal lock |beta| = 90)', () => {
    for (let i = 0; i < 2000; i++) {
      const alpha = rng() * 360;
      let beta = rng() * 360 - 180;
      if (Math.abs(Math.abs(beta) - 90) < 1) beta += 2;
      const gamma = rng() * 179.8 - 89.9;
      const e = eulerZXYFromMat(matFromEulerZXY(alpha, beta, gamma));
      expectInW3cRange(e);
      expect(Math.abs(angDiff(e.alpha, alpha))).toBeLessThan(1e-8);
      expect(Math.abs(angDiff(e.beta, beta))).toBeLessThan(1e-8);
      expect(Math.abs(e.gamma - gamma)).toBeLessThan(1e-8);
    }
  });

  it('mat → euler → mat round trip for random rotations (always valid)', () => {
    for (let i = 0; i < 5000; i++) {
      const r = randRotation(rng);
      const e = eulerZXYFromMat(r);
      expectInW3cRange(e);
      expectMatClose(matFromEulerZXY(e.alpha, e.beta, e.gamma), r, 1e-8);
    }
  });

  it('edge cases: exact gimbal lock, gamma = −90, beta = −180, gamma rounding to +90', () => {
    const cases: Mat3[] = [
      [1, 0, 0, 0, 0, -1, 0, 1, 0], // Rx(+90) exactly: m20 = m22 = 0 (gimbal lock)
      [1, 0, 0, 0, 0, 1, 0, -1, 0], // Rx(−90) exactly
      mMul(mAxisAngle([0, 0, 1], 0.7), [1, 0, 0, 0, 0, -1, 0, 1, 0]), // gimbal with a heading
      [0, 0, -1, 0, 1, 0, 1, 0, 0], // Ry(−90) exactly: m22 = 0, m20 > 0
      mMul([-1, 0, 0, 0, -1, 0, 0, 0, 1], [0, 0, -1, 0, 1, 0, 1, 0, 0]), // m22 = 0 with cos(beta) < 0
      [1, 0, 0, 0, -1, 0, 0, 0, -1], // Rx(180): beta = −180, not +180
      matFromEulerZXY(10, 20, 90), // gamma = +90 is outside the W3C range
      matFromEulerZXY(200, -150, 90),
      matFromEulerZXY(0, 90, 0),
      matFromEulerZXY(123, -90, 0),
      matFromEulerZXY(359.9999999999, 0, 0),
      matFromEulerZXY(-1e-13, 0, 0),
    ];
    for (const m of cases) {
      const e = eulerZXYFromMat(m);
      expectInW3cRange(e);
      expectMatClose(matFromEulerZXY(e.alpha, e.beta, e.gamma), m, 1e-9);
    }
    expect(eulerZXYFromMat([1, 0, 0, 0, 0, -1, 0, 1, 0])).toEqual({ alpha: 0, beta: 90, gamma: 0 });
    expect(eulerZXYFromMat([1, 0, 0, 0, -1, 0, 0, 0, -1])).toEqual({ alpha: 0, beta: -180, gamma: 0 });
    expect(eulerZXYFromMat([0, 0, -1, 0, 1, 0, 1, 0, 0])).toEqual({ alpha: 0, beta: 0, gamma: -90 });
  });
});

describe('rotationRate mapping', () => {
  it('W3C 2025: alpha ↔ x, beta ↔ y, gamma ↔ z (not the Z-X′-Y″ order of the angles)', () => {
    expect(rotationRateToVector({ alpha: 1, beta: 2, gamma: 3 })).toEqual([1, 2, 3]);
    expect(vectorToRotationRate([1, 2, 3])).toEqual({ alpha: 1, beta: 2, gamma: 3 });
    const w: Vec3 = [-4, 5, 6];
    expect(rotationRateToVector(vectorToRotationRate(w))).toEqual(w);
  });
});
