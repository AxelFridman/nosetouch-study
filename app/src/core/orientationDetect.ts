/**
 * Infers how the phone is rotated from the gravity direction in the device frame.
 *
 * `accg` is a mean accelerationIncludingGravity vector (raw device frame). With the W3C convention
 * (gravitySign = +1) a phone at rest reports the reaction to gravity, i.e. accg points UP:
 *   upright (top edge up)        → accg ≈ (0, +g, 0)
 *   upside_down                  → accg ≈ (0, −g, 0)
 *   top_left (top edge to the participant's left, screen toward or away)
 *        screen toward: device +x points up → accg ≈ (+g, 0, 0)
 *        screen away:   device +x points down → accg ≈ (−g, 0, 0)   (see rotationFromGravity docs)
 *   top_right: the opposite x signs.
 *   flat (screen up)             → accg ≈ (0, 0, +g)
 * When gravitySign = −1 the vector is negated first; when null, +1 is assumed.
 * Screen toward/away cannot be told from gravity for upright/upside-down poses; for landscape it is
 * coupled with top_left/top_right, so the classifier returns the rotation *given* the instructed
 * screen facing (`screenHint`). An axis must dominate (angle to it < maxAngleDeg) or 'unknown'.
 *
 * The table matches the grip matrices of sim/model.ts (accg = gravitySign·g·(R_BD columns)·ŷ_body).
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import type { PhoneRotation, ScreenFacing, Vec3 } from './types';

const RAD2DEG = 180 / Math.PI;

function isVec3(v: Vec3 | null | undefined): v is Vec3 {
  return !!v && Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

/**
 * The dominant axis is the component with the largest magnitude; its signed direction must be
 * within maxAngleDeg (strictly less) of the vector. Mapping (after applying gravitySign):
 *   +y upright · −y upside_down · +z flat · −z (screen facing the floor) 'unknown': no such PhoneRotation
 *   ±x: screenHint 'away' → +x top_right, −x top_left; 'toward', 'up' or null (assumed toward,
 *       the default grip) → +x top_left, −x top_right.
 * Zero, missing or non-finite vectors → 'unknown'.
 */
export function rotationFromGravity(
  accg: Vec3,
  gravitySign: 1 | -1 | null,
  screenHint: ScreenFacing | null,
  maxAngleDeg?: number, // default 35
): PhoneRotation | 'unknown' {
  if (!isVec3(accg)) return 'unknown';
  const maxAngle = maxAngleDeg ?? 35;
  const sign = gravitySign === -1 ? -1 : 1;
  const x = sign * accg[0];
  const y = sign * accg[1];
  const z = sign * accg[2];
  const n = Math.sqrt(x * x + y * y + z * z);
  if (!(n > 0) || !Number.isFinite(n)) return 'unknown';
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  let axis: 'x' | 'y' | 'z';
  let comp: number;
  if (ax >= ay && ax >= az) {
    axis = 'x';
    comp = x;
  } else if (ay >= az) {
    axis = 'y';
    comp = y;
  } else {
    axis = 'z';
    comp = z;
  }
  const angle = Math.acos(Math.min(1, Math.abs(comp) / n)) * RAD2DEG;
  if (!(angle < maxAngle)) return 'unknown';
  if (axis === 'y') return comp > 0 ? 'upright' : 'upside_down';
  if (axis === 'z') return comp > 0 ? 'flat' : 'unknown';
  const away = screenHint === 'away';
  if (comp > 0) return away ? 'top_right' : 'top_left';
  return away ? 'top_left' : 'top_right';
}

/** Angle (deg) between two gravity vectors; null if either is missing/zero. */
export function tiltChangeDeg(a: Vec3 | null, b: Vec3 | null): number | null {
  if (!isVec3(a) || !isVec3(b)) return null;
  const na = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const nb = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  if (!(na > 0) || !(nb > 0) || !Number.isFinite(na) || !Number.isFinite(nb)) return null;
  // atan2(|a×b|, a·b) stays accurate near 0° and 180°, where acos of a rounded cosine does not.
  const cx = a[1] * b[2] - a[2] * b[1];
  const cy = a[2] * b[0] - a[0] * b[2];
  const cz = a[0] * b[1] - a[1] * b[0];
  const cross = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return Math.atan2(cross, dot) * RAD2DEG;
}
