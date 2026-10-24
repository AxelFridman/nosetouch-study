/**
 * How a guided calibration window is judged (pure). The v1 calibration screen used these to explain
 * its results; v2 calibrates implicitly from the trials (spec §5) and has no calibration screen, but
 * the rules stay tested against core/calibration.ts (tests/calibration.test.ts), and a future guided
 * check can reuse them. The math itself lives in core/calibration.ts.
 */
import type { AxisCheck, CalibrationResult, PoseStats } from '../core/types';

/** 'face_down': a sign the engine expectation contradicts (|z| ≈ |g| also when lying screen-down). */
export type FlatVerdict = 'ok' | 'ok_inverted' | 'face_down' | 'moving' | 'not_flat' | 'odd';
export type UprightVerdict = 'ok' | 'inverted' | 'moving' | 'unclear';
export type AxisVerdict = 'ok' | 'noted' | 'radians' | 'unclear';

/**
 * Why a flat measurement did (not) give a usable gravity sign — mirrors the gravityFromFlat rules.
 * `expected`: engineGravitySign(device.engine); null (unknown engine, simulation) accepts either sign.
 */
export function flatVerdict(stats: PoseStats | null, sign: 1 | -1 | null, expected: 1 | -1 | null = null): FlatVerdict | null {
  if (!stats) return null;
  if (sign !== null && expected !== null && sign !== expected) return 'face_down';
  if (sign === 1) return 'ok';
  if (sign === -1) return 'ok_inverted';
  if (!stats.still) return 'moving';
  const [x, y, z] = stats.accg_mean;
  const g = Math.hypot(x, y, z);
  if (!(g >= 8.6 && g <= 11)) return 'odd';
  return Math.abs(z) < 0.9 * g ? 'not_flat' : 'odd';
}

export function uprightVerdict(stats: PoseStats | null, check: CalibrationResult['upright_check']): UprightVerdict | null {
  if (!stats) return null;
  if (check === 'ok') return 'ok';
  if (check === 'inverted') return 'inverted';
  return stats.still ? 'unclear' : 'moving';
}

export function axisVerdict(a: AxisCheck | null): AxisVerdict | null {
  if (!a) return null;
  if (a.verdict === 'unclear') return 'unclear';
  if (a.units === 'rad_per_s') return 'radians';
  return a.verdict === 'w3c' ? 'ok' : 'noted';
}
