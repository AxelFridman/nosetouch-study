import { describe, expect, it } from 'vitest';
import { ORIENTATIONS } from '../src/core/constants';
import { rotationFromGravity, tiltChangeDeg } from '../src/core/orientationDetect';
import type { OrientationId, Vec3 } from '../src/core/types';
import { G, rotateInv, segRng } from './helpers/segSignals';

/**
 * Grip matrices R_BD (columns = device x, y, z axes in body coordinates; body x = participant's
 * right, y = up, z = backward), copied from the sim/model.ts contract so this test stays
 * independent of the simulator implementation.
 */
const GRIP: Record<Exclude<OrientationId, 'free'>, [Vec3, Vec3, Vec3]> = {
  upright_toward: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  upright_away: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
  top_left_toward: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
  top_right_toward: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  upside_down_toward: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
  upside_down_away: [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
  top_left_away: [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
  top_right_away: [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
  flat_up: [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
};

/** accg at rest = gravitySign · R_BDᵀ · (0, g, 0) = gravitySign · g · (c_x·ŷ, c_y·ŷ, c_z·ŷ). */
function restingAccg(o: Exclude<OrientationId, 'free'>, sign: 1 | -1): Vec3 {
  const [cx, cy, cz] = GRIP[o];
  return [sign * G * cx[1], sign * G * cy[1], sign * G * cz[1]];
}

const ORIENTS = Object.keys(GRIP) as Exclude<OrientationId, 'free'>[];

describe('rotationFromGravity', () => {
  it('the grip matrices are proper rotations (sanity check of the copied table)', () => {
    for (const o of ORIENTS) {
      const [a, b, c] = GRIP[o];
      const det =
        a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) + c[0] * (a[1] * b[2] - a[2] * b[1]);
      expect(det, o).toBe(1);
    }
  });

  for (const sign of [1, -1] as const) {
    it(`recognises every instructed orientation at rest (gravitySign ${sign})`, () => {
      for (const o of ORIENTS) {
        const info = ORIENTATIONS[o];
        expect(rotationFromGravity(restingAccg(o, sign), sign, info.screen), o).toBe(info.rotation);
      }
    });

    it(`tolerates a 25° grip tilt about random axes but not 40° (gravitySign ${sign})`, () => {
      const rng = segRng(sign === 1 ? 11 : 12);
      for (const o of ORIENTS) {
        const info = ORIENTATIONS[o];
        for (let k = 0; k < 20; k++) {
          const axis: Vec3 = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
          const n = Math.hypot(...axis);
          const u: Vec3 = [axis[0] / n, axis[1] / n, axis[2] / n];
          const g = restingAccg(o, sign);
          // tilting about an axis ⟂ gravity by φ moves gravity by exactly φ
          const gn = Math.hypot(...g);
          const d: Vec3 = [g[0] / gn, g[1] / gn, g[2] / gn];
          const dot = u[0] * d[0] + u[1] * d[1] + u[2] * d[2];
          const perp: Vec3 = [u[0] - dot * d[0], u[1] - dot * d[1], u[2] - dot * d[2]];
          const pn = Math.hypot(...perp);
          const w: Vec3 = [perp[0] / pn, perp[1] / pn, perp[2] / pn];
          expect(rotationFromGravity(rotateInv(w, 25, g), sign, info.screen), `${o} 25°`).toBe(info.rotation);
          expect(rotationFromGravity(rotateInv(w, 40, g), sign, info.screen), `${o} 40°`).toBe('unknown');
        }
      }
    });
  }

  it('a wrong gravity sign turns portrait upside down and swaps the landscape sides', () => {
    expect(rotationFromGravity(restingAccg('upright_toward', -1), 1, 'toward')).toBe('upside_down');
    expect(rotationFromGravity(restingAccg('top_left_toward', -1), 1, 'toward')).toBe('top_right');
    expect(rotationFromGravity(restingAccg('flat_up', -1), 1, 'up')).toBe('unknown');
  });

  it('null gravitySign means +1 (W3C)', () => {
    for (const o of ORIENTS) {
      const info = ORIENTATIONS[o];
      expect(rotationFromGravity(restingAccg(o, 1), null, info.screen), o).toBe(info.rotation);
    }
  });

  it("landscape: without a screen hint (or hint 'up') the phone is assumed to face the participant", () => {
    expect(rotationFromGravity([G, 0, 0], 1, null)).toBe('top_left');
    expect(rotationFromGravity([-G, 0, 0], 1, null)).toBe('top_right');
    expect(rotationFromGravity([G, 0, 0], 1, 'up')).toBe('top_left');
    expect(rotationFromGravity([G, 0, 0], 1, 'away')).toBe('top_right');
    expect(rotationFromGravity([-G, 0, 0], 1, 'away')).toBe('top_left');
    // portrait does not depend on the hint
    expect(rotationFromGravity([0, G, 0], 1, 'away')).toBe('upright');
    expect(rotationFromGravity([0, -G, 0], 1, null)).toBe('upside_down');
  });

  it("screen facing the floor, an undecided vector, or no vector at all → 'unknown'", () => {
    expect(rotationFromGravity([0, 0, -G], 1, 'up')).toBe('unknown');
    expect(rotationFromGravity([0, G, G], 1, 'toward')).toBe('unknown'); // 45° between two axes
    expect(rotationFromGravity([0, 0, 0], 1, 'toward')).toBe('unknown');
    expect(rotationFromGravity([Number.NaN, G, 0], 1, 'toward')).toBe('unknown');
    expect(rotationFromGravity(null as unknown as Vec3, 1, 'toward')).toBe('unknown');
  });

  it('respects maxAngleDeg (strictly less than)', () => {
    const tilt30 = rotateInv([1, 0, 0], 30, [0, G, 0]); // exactly 30° from +y
    expect(rotationFromGravity(tilt30, 1, 'toward')).toBe('upright');
    expect(rotationFromGravity(tilt30, 1, 'toward', 31)).toBe('upright');
    expect(rotationFromGravity(tilt30, 1, 'toward', 29)).toBe('unknown');
    expect(rotationFromGravity(rotateInv([1, 0, 0], 44, [0, G, 0]), 1, 'toward', 50)).toBe('upright');
    // works for any magnitude (only the direction matters)
    expect(rotationFromGravity([0, 0.01, 0], 1, 'toward')).toBe('upright');
  });
});

describe('tiltChangeDeg', () => {
  it('is the angle between the two vectors, independent of their lengths', () => {
    expect(tiltChangeDeg([0, G, 0], [0, G, 0])).toBe(0);
    expect(tiltChangeDeg([0, G, 0], [G, 0, 0])).toBeCloseTo(90, 12);
    expect(tiltChangeDeg([0, 1, 0], [0, -5, 0])).toBeCloseTo(180, 12);
    expect(tiltChangeDeg([0, 1, 0], [0, 3, 3])).toBeCloseTo(45, 12);
    expect(tiltChangeDeg([1, 2, 3], [-2, 1, 0])).toBeCloseTo(90, 12);
    // accurate for tiny angles (acos of a rounded cosine would give 0 or ~1e-6°)
    const tiny = rotateInv([0, 0, 1], 1e-7, [0, G, 0]);
    expect(tiltChangeDeg([0, G, 0], tiny)).toBeCloseTo(1e-7, 12);
  });

  it('is null when a vector is missing, zero or not finite', () => {
    expect(tiltChangeDeg(null, [0, 1, 0])).toBeNull();
    expect(tiltChangeDeg([0, 1, 0], null)).toBeNull();
    expect(tiltChangeDeg([0, 0, 0], [0, 1, 0])).toBeNull();
    expect(tiltChangeDeg([0, 1, 0], [Number.NaN, 1, 0])).toBeNull();
    expect(tiltChangeDeg([0, 1, 0], [Number.POSITIVE_INFINITY, 1, 0])).toBeNull();
  });
});
