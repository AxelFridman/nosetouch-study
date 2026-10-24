/**
 * Kinematic model of a person bringing a phone from an extended arm to the nose.
 *
 * Frames
 *  Body B (origin at the nose tip): x = participant's right, y = up, z = backward (face → back of
 *  head). The extended-arm phone is at z ≈ −0.55 m.
 *  Earth E (W3C): x = East, y = North, z = Up. A participant facing heading ψ (deg clockwise from
 *  North): R_EB columns = [right, up, backward] = [(cosψ, −sinψ, 0), (0, 0, 1), (−sinψ, −cosψ, 0)].
 *  Device D (W3C): x right edge, y top edge, z out of the screen.
 *  Grip R_BD(o) (columns = device axes in body coords), e.g. upright_toward = I,
 *  upright_away = [(−1,0,0),(0,1,0),(0,0,−1)], top_left_toward = [(0,1,0),(−1,0,0),(0,0,1)],
 *  top_right_toward = [(0,−1,0),(1,0,0),(0,0,1)], upside_down_toward = [(−1,0,0),(0,−1,0),(0,0,1)],
 *  upside_down_away = [(1,0,0),(0,−1,0),(0,0,−1)], top_left_away = [(0,−1,0),(−1,0,0),(0,0,−1)],
 *  top_right_away = [(0,1,0),(1,0,0),(0,0,−1)], flat_up = [(1,0,0),(0,0,−1),(0,1,0)];
 *  'free' = a random one of the above (uniform; the LEFT hand takes the mirror of the drawn one so
 *  that left/right stay exact mirror images) — except in street's 'choice' trial, where the
 *  participant picks their own hand and the natural upright grip (performedCondition, applied by
 *  the simulated source). Plus a small random per-trial grip tilt (≤ 10°),
 *  applied in the body frame: R_grip = exp([τ]×)·R_BD(o).
 *  Mirror property: M·R_BD(o)·S = R_BD(mirror(o)) with M = S = diag(−1,1,1) — tests must check it.
 *
 * Movement (h = +1 right hand, −1 left hand; σ(s) = 10s³ − 15s⁴ + 6s⁵ minimum jerk, s = t/T):
 *  progress u = σ(s) (variants may chain several minimum-jerk pieces, see below)
 *  position p(u) = p0 + (p1 − p0)·u + arc·sin(πu)·n̂, n̂ = normalize((h·0.6, 0.8, 0)) (out & up)
 *    p0 shoulder = (h·w, −0.24, 0.08 − L)    (w = shoulder half-width, L = arm length)
 *    p0 center   = (h·0.03, −0.12, −(L − 0.08))
 *    p0 natural  = per-participant blend between shoulder and center
 *    p1 (nose contact) = (0, −0.03, −0.01) + small noise
 *    (the start height varies ±3 cm per trial, the same offset for every start position)
 *  rotation R_BD(t) = R_motion(u) · R_grip, R_motion(u) = Ry(h·yaw(u)) · Rx(pitch·u) · Rz(h·roll·u),
 *    yaw(u) = yaw_net·u + yaw_transient·sin(πu)  (degrees, converted to rad)
 *  NOTE: the bump terms use sin(π·u) — not sin(π·s) — so that velocity and acceleration (linear
 *  and angular) are continuous and zero at both ends; sin(πs) would start the movement with a
 *  velocity step that the analytic acceleration never shows.
 *  This makes the left-hand movement the exact mirror image of the right-hand one; asymmetry comes
 *  only from participant dominance (non-dominant hand: slower, noisier) — a realistic confound.
 *  Every random lateral quantity is drawn ONCE and multiplied by h (x of polar vectors such as
 *  positions; y and z of axial vectors such as the grip tilt), and the draws happen in a fixed
 *  order whatever the hand/variant, so equal rng streams give exactly mirrored movements.
 *  Speeds: natural T ≈ style.duration_natural_s, fast ≈ 0.55·natural (but ≥ 0.4 s, a physiological
 *  floor for a ~0.6 m reach), slow ≈ 2.0·natural (the Android pilot's clean trials: slow onset-to-contact
 *  1.95–2.55 s vs natural 0.97–1.21 s; 2.6 gave slow peaks well below the pilot's ≥ 50 deg/s and missed
 *  ~2 % of slow onsets under protocol 2.0.0's slow floors); then ±12% per trial (× nondominant_jitter).
 *  Variants modify the plan: curved (arc ×3), pause (σ holds at 0.5 for 0.8 s: two half-way
 *  minimum-jerk pieces with a still plateau between), twist (roll ×3, yaw_transient ×2),
 *  elbow_high (pitch −25°, arc ×1.5 upward), bent_start (L ×0.8), lean_in (p1 z −0.08, T ×0.9),
 *  chin (p1 y −0.07), forehead (p1 y +0.06), jerky (3 stop-go minimum-jerk steps, 90–200 ms stops),
 *  tilted_touch (the top edge touches the nose: an extra rotation about the DEVICE x axis,
 *  R_BD(u) = R_motion(u)·R_grip·Rx(θ·u²), so the top edge tips towards the face as the phone arrives;
 *  θ = PlanOptions.tiltedTouchDeg, default 35°; a device-x rotation is its own mirror image, so the
 *  same θ for both hands keeps the mirror property).
 *
 * Signals at time t (all exact, then noise is added by the source):
 *  a_B = d²p/dt² (analytic), specific force f_B = a_B + (0, g, 0)
 *  accg_D = gravitySign · R_BDᵀ f_B ;  acc_D = gravitySign · R_BDᵀ a_B
 *  ω_B from mLog(R_BD(t+h)·R_BD(t−h)ᵀ)/(2h); ω_D = R_BDᵀ ω_B; rr = vectorToRotationRate(ω_D in deg/s)
 *  orientation = eulerZXYFromMat(R_EB · R_BD); quaternion = quatFromMat(R_EB · R_BD)
 *
 * Randomness: this module never uses Math.random. simRng() is the same mulberry32 algorithm as
 * core/util.makeRng, duplicated here so the simulator (and its tests) have no runtime dependency
 * on other modules.
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { GRAVITY, ORIENTATIONS } from '../core/constants';
import { DEG, mExp, mFromColumns, mMul, vNormalize, type Mat3 } from '../core/math3d';
import type {
  Hand,
  HandednessSelf,
  OrientationId,
  ParticipantProfile,
  PhoneHand,
  Sex,
  TrialCondition,
  Vec3,
} from '../core/types';

export interface SimStyle {
  duration_natural_s: number;
  yaw_net_deg: number;
  yaw_transient_deg: number;
  roll_deg: number;
  pitch_deg: number;
  arc_m: number;
  start_blend: number; // 0 = shoulder line, 1 = centered, used for start_position 'natural'
  reaction_ms: number;
  tremor_dps: number;
  nondominant_slowdown: number; // multiplies T for the non-dominant hand
  nondominant_jitter: number; // multiplies noise/variability for the non-dominant hand
}

export interface SimParticipant {
  participant_id: string;
  sex: Sex;
  age_years: number;
  height_cm: number;
  handedness: HandednessSelf;
  dominant: Hand;
  arm_length_m: number;
  shoulder_half_width_m: number;
  style: SimStyle;
  /**
   * The hand that usually holds the phone (profile.phone_hand). Used where the participant chooses
   * the hand: street's 'choice' trial (performedCondition). Absent or 'both' → the dominant hand.
   */
  phone_hand?: PhoneHand;
}

export interface SimPlatform {
  name: 'ios' | 'android';
  gravitySign: 1 | -1;
  hz: number;
  jitter_ms: number;
  drop_prob: number;
  acc_available: boolean;
  quaternion: boolean;
  acc_noise_ms2: number;
  gyro_noise_dps: number;
  gyro_bias_dps: Vec3;
  heading_deg: number;
  /**
   * rotationRate field mapping. 'w3c' (default): (alpha, beta, gamma) = rates about device (x, y, z),
   * as every current browser reports. 'legacy': the pre-2019 reading (alpha about z), for testing
   * the calibration axis check and the analysis normalisation layer.
   */
  rr_mapping?: 'w3c' | 'legacy';
  /**
   * Chromium's privacy rounding (platform_sensor_util.cc, crbug 1018180): acc/accg to multiples of
   * 0.1 m/s², rotationRate to 0.1 deg/s (rounded in rad/s, then converted, as Blink does),
   * deviceorientation angles to 0.1° and quaternions to 0.1° via axis-angle; nearest multiple,
   * halves away from zero. WebKit does not round. Undefined → false. A source created with
   * noise: false ignores it (exact tests).
   */
  quantize?: boolean;
  /**
   * Chrome Android's magnetometer-fused, heading-referenced streams (schema 1.2): an
   * AbsoluteOrientationSensor quaternion next to every relative one (needs `quaternion`) and a
   * 'deviceorientationabsolute' event next to every deviceorientation event. Undefined → false.
   */
  absolute_orientation?: boolean;
}

/** Kinematic state at one instant: body-frame position/velocity/acceleration and R_BD. */
export interface PoseState {
  p: Vec3;
  v: Vec3;
  a: Vec3;
  R: Mat3;
}

export interface MovementPlan {
  hand: Hand;
  orientation: Exclude<OrientationId, 'free'>;
  duration_s: number;
  p0: Vec3;
  p1: Vec3;
  grip: Mat3; // R_BD at rest (includes the random grip tilt)
  /** evaluates the movement at s = t/T (clamped to [0,1] outside) */
  pose(tSinceOnset_s: number): { p: Vec3; v: Vec3; a: Vec3; R: Mat3 };
  /** the geometric path at progress u ∈ [0,1] with time derivatives du, ddu (used for the return) */
  path(u: number, du: number, ddu: number): PoseState;
  /** +1 right hand, −1 left hand */
  h: 1 | -1;
  nondominant: boolean;
  /** peak of the 60 ms contact transient at the nose (m/s²; ≤ 3, largest for fast movements) */
  contact_ms2: number;
}

export type GripOrientation = Exclude<OrientationId, 'free'>;

/** The nine concrete grips; 'free' draws uniformly from this list. */
export const GRIP_ORIENTATIONS: GripOrientation[] = [
  'upright_toward',
  'upright_away',
  'top_left_toward',
  'top_right_toward',
  'upside_down_toward',
  'upside_down_away',
  'top_left_away',
  'top_right_away',
  'flat_up',
];

/**
 * Sign of acceleration / accelerationIncludingGravity on simulated iOS devices. WebKit forwards
 * Core Motion values without negation, so a face-up iPhone reads accg_z ≈ −9.81 where the W3C
 * convention (Android) reads +9.81 (WebKit WebCoreMotionManager.mm; re-measured on iPadOS 26.6 in
 * Oct 2026). rotationRate signs agree across platforms. scripts/synth.ts can override it.
 */
export const IOS_GRAVITY_SIGN: 1 | -1 = -1;

/** Gravity reaction expressed in the body frame (what an accelerometer at rest measures, W3C sign). */
export const BODY_GRAVITY: Vec3 = [0, GRAVITY, 0];

// ───────────────────────────── randomness ─────────────────────────────

/** mulberry32 (same algorithm as core/util.makeRng): deterministic uint32-seeded PRNG in [0, 1). */
export function simRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Derives an independent uint32 seed (so separate random streams never share draws). */
export function deriveSeed(seed: number, stream: number): number {
  let h = (seed ^ Math.imul(stream + 1, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

/** Standard normal (Box–Muller). Always consumes exactly two draws, which keeps streams aligned. */
export function gaussian(rng: () => number): number {
  const u1 = 1 - rng(); // (0, 1]
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

// ───────────────────────────── minimum jerk ─────────────────────────────

/** Minimum-jerk profile σ(s) and its derivatives with respect to s (clamped outside [0, 1]). */
export function minJerk(s: number): { u: number; du: number; ddu: number } {
  if (s <= 0) return { u: 0, du: 0, ddu: 0 };
  if (s >= 1) return { u: 1, du: 0, ddu: 0 };
  const s2 = s * s;
  const s3 = s2 * s;
  return {
    u: 10 * s3 - 15 * s3 * s + 6 * s3 * s2,
    du: 30 * s2 - 60 * s3 + 30 * s2 * s2,
    ddu: 60 * s - 180 * s2 + 120 * s3,
  };
}

/** One minimum-jerk piece of the progress variable: u goes u0 → u1 during [t0, t0 + dur] (seconds). */
interface ProgressPiece {
  t0: number;
  dur: number;
  u0: number;
  u1: number;
}

function progressAt(pieces: ProgressPiece[], t: number): { u: number; du: number; ddu: number } {
  let u = 0;
  for (const pc of pieces) {
    if (t < pc.t0) return { u, du: 0, ddu: 0 }; // still plateau before this piece
    if (t <= pc.t0 + pc.dur) {
      const m = minJerk((t - pc.t0) / pc.dur);
      const span = pc.u1 - pc.u0;
      return { u: pc.u0 + span * m.u, du: (span * m.du) / pc.dur, ddu: (span * m.ddu) / (pc.dur * pc.dur) };
    }
    u = pc.u1;
  }
  return { u, du: 0, ddu: 0 };
}

// ───────────────────────────── frames ─────────────────────────────

const GRIPS: Record<GripOrientation, [Vec3, Vec3, Vec3]> = {
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

export function gripMatrix(o: Exclude<OrientationId, 'free'>): Mat3 {
  const [c0, c1, c2] = GRIPS[o];
  return mFromColumns(c0, c1, c2);
}

export function earthFromBody(headingDeg: number): Mat3 {
  const c = Math.cos(headingDeg * DEG);
  const s = Math.sin(headingDeg * DEG);
  return mFromColumns([c, -s, 0], [0, 0, 1], [-s, -c, 0]);
}

function rotX(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}
function rotY(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
function rotZ(rad: number): Mat3 {
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

// ───────────────────────────── movement plan ─────────────────────────────

const SPEED_FACTOR = { natural: 1, fast: 0.55, slow: 2.0 } as const;
const FAST_FLOOR_S = 0.4;
const PAUSE_S = 0.8;
const MAX_TILT_DEG = 10;
/** tilted_touch: default end tilt about the device x axis (the instruction asks for ~30–40°). */
export const TILTED_TOUCH_DEFAULT_DEG = 35;

/** Plan parameters drawn outside planMovement's own random stream (so its draws never shift). */
export interface PlanOptions {
  /** tilted_touch: end tilt about the device x axis, deg (default TILTED_TOUCH_DEFAULT_DEG) */
  tiltedTouchDeg?: number;
}

export function planMovement(p: SimParticipant, c: TrialCondition, rng: () => number, planOpts: PlanOptions = {}): MovementPlan {
  const st = p.style;
  const h: 1 | -1 = c.hand === 'right' ? 1 : -1;
  const nondominant = c.hand !== p.dominant;
  const jf = nondominant ? st.nondominant_jitter : 1; // variability multiplier
  const slowdown = nondominant ? st.nondominant_slowdown : 1;

  // ── all draws, in a fixed order that never depends on hand, speed or variant ──
  const uOrient = rng();
  const uDur = rng();
  const tiltDir: Vec3 = [gaussian(rng), gaussian(rng), gaussian(rng)];
  const uTilt = rng();
  const zP1: Vec3 = [gaussian(rng), gaussian(rng), gaussian(rng)];
  const zYaw = gaussian(rng);
  const zTrans = gaussian(rng);
  const zRoll = gaussian(rng);
  const zPitch = gaussian(rng);
  const zArc = gaussian(rng);
  const zBlend = gaussian(rng);
  const zStartY = gaussian(rng);
  const zReach = gaussian(rng);
  const uSteps = [rng(), rng(), rng()]; // jerky: relative step sizes
  const uStepDur = [rng(), rng(), rng()]; // jerky: relative step durations
  const uStops = [rng(), rng()]; // jerky: stop durations

  // 'free': uniform over the nine grips; the left hand takes the mirror so pairs stay mirrored.
  let orientation: GripOrientation;
  if (c.orientation === 'free') {
    const drawn = GRIP_ORIENTATIONS[Math.min(GRIP_ORIENTATIONS.length - 1, Math.floor(uOrient * GRIP_ORIENTATIONS.length))];
    orientation = h === 1 ? drawn : (ORIENTATIONS[drawn].mirror as GripOrientation);
  } else {
    orientation = c.orientation;
  }

  // grip tilt: body-frame rotation vector, axial → y and z flip with the hand
  const tiltUnit = vNormalize(tiltDir);
  const tiltRad = Math.min(MAX_TILT_DEG, 7 * jf * uTilt) * DEG;
  const tau: Vec3 = [tiltUnit[0] * tiltRad, h * tiltUnit[1] * tiltRad, h * tiltUnit[2] * tiltRad];
  const grip = mMul(mExp(tau), gripMatrix(orientation));

  // duration; a ~0.6 m reach cannot be made much faster than 0.4 s, whatever the natural pace
  const base = c.speed === 'fast' ? Math.max(FAST_FLOOR_S, st.duration_natural_s * SPEED_FACTOR.fast) : st.duration_natural_s * SPEED_FACTOR[c.speed];
  let T = base * slowdown * (1 + 0.12 * jf * (2 * uDur - 1));
  if (c.variant === 'lean_in') T *= 0.9;
  T = Math.max(0.25, T);

  // start position
  const L = p.arm_length_m * (c.variant === 'bent_start' ? 0.8 : 1) * (1 + 0.03 * jf * clamp(zReach, -2.5, 2.5));
  const dy = 0.03 * jf * clamp(zStartY, -2.5, 2.5);
  const shoulder: Vec3 = [h * p.shoulder_half_width_m, -0.24 + dy, 0.08 - L];
  const center: Vec3 = [h * 0.03, -0.12 + dy, -(L - 0.08)];
  let p0: Vec3;
  if (c.start_position === 'shoulder') p0 = shoulder;
  else if (c.start_position === 'center') p0 = center;
  else {
    const b = clamp(st.start_blend + 0.08 * jf * zBlend, 0, 1);
    p0 = [shoulder[0] + (center[0] - shoulder[0]) * b, shoulder[1] + (center[1] - shoulder[1]) * b, shoulder[2] + (center[2] - shoulder[2]) * b];
  }

  // nose contact point (polar → x flips with the hand)
  const p1: Vec3 = [
    h * 0.006 * jf * clamp(zP1[0], -3, 3),
    -0.03 + 0.006 * jf * clamp(zP1[1], -3, 3),
    -0.01 + 0.003 * jf * clamp(zP1[2], -3, 3),
  ];
  if (c.variant === 'lean_in') p1[2] -= 0.08;
  if (c.variant === 'chin') p1[1] -= 0.07;
  if (c.variant === 'forehead') p1[1] += 0.06;

  // path shape
  let arc = Math.max(0, st.arc_m * (1 + 0.2 * jf * clamp(zArc, -2.5, 2.5)));
  if (c.variant === 'curved') arc *= 3;
  if (c.variant === 'elbow_high') arc *= 1.5;
  const nHat = vNormalize(c.variant === 'elbow_high' ? [h * 0.3, 0.95, 0] : [h * 0.6, 0.8, 0]);

  // rotation amplitudes (deg)
  const yawNet = st.yaw_net_deg + 4 * jf * zYaw;
  const yawTr = st.yaw_transient_deg * (c.variant === 'twist' ? 2 : 1) + 3 * jf * zTrans;
  const roll = st.roll_deg * (c.variant === 'twist' ? 3 : 1) + 4 * jf * zRoll;
  const pitch = st.pitch_deg + (c.variant === 'elbow_high' ? -25 : 0) + 5 * jf * zPitch;

  // progress pieces
  let pieces: ProgressPiece[];
  if (c.variant === 'pause') {
    pieces = [
      { t0: 0, dur: T / 2, u0: 0, u1: 0.5 },
      { t0: T / 2 + PAUSE_S, dur: T / 2, u0: 0.5, u1: 1 },
    ];
  } else if (c.variant === 'jerky') {
    const sizes = uSteps.map((v) => 0.6 + 0.8 * v);
    const sizeSum = sizes[0] + sizes[1] + sizes[2];
    const durs = uStepDur.map((v) => 0.8 + 0.4 * v);
    const durSum = durs[0] + durs[1] + durs[2];
    const stops = uStops.map((v) => 0.09 + 0.11 * v);
    pieces = [];
    let t = 0;
    let u = 0;
    for (let i = 0; i < 3; i++) {
      const dur = (T * durs[i]) / durSum;
      const u1 = i === 2 ? 1 : u + sizes[i] / sizeSum;
      pieces.push({ t0: t, dur, u0: u, u1 });
      t += dur + (i < 2 ? stops[i] : 0);
      u = u1;
    }
  } else {
    pieces = [{ t0: 0, dur: T, u0: 0, u1: 1 }];
  }
  const last = pieces[pieces.length - 1];
  const duration = last.t0 + last.dur;

  const delta: Vec3 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const tiltTouchRad =
    c.variant === 'tilted_touch'
      ? (Number.isFinite(planOpts.tiltedTouchDeg) ? (planOpts.tiltedTouchDeg as number) : TILTED_TOUCH_DEFAULT_DEG) * DEG
      : 0;

  const path = (u: number, du: number, ddu: number): PoseState => {
    const sinU = Math.sin(Math.PI * u);
    const cosU = Math.cos(Math.PI * u);
    const pos: Vec3 = [0, 0, 0];
    const vel: Vec3 = [0, 0, 0];
    const acc: Vec3 = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const dpdu = delta[i] + arc * Math.PI * cosU * nHat[i];
      const d2pdu2 = -arc * Math.PI * Math.PI * sinU * nHat[i];
      pos[i] = p0[i] + delta[i] * u + arc * sinU * nHat[i];
      vel[i] = dpdu * du;
      acc[i] = d2pdu2 * du * du + dpdu * ddu;
    }
    const yaw = (yawNet * u + yawTr * sinU) * DEG;
    const rMotion = mMul(mMul(rotY(h * yaw), rotX(pitch * u * DEG)), rotZ(h * roll * u * DEG));
    const R = mMul(rMotion, grip);
    // tilted_touch: device-frame rotation about x (right-multiplied), growing as u² towards the nose
    return { p: pos, v: vel, a: acc, R: tiltTouchRad === 0 ? R : mMul(R, rotX(tiltTouchRad * u * u)) };
  };

  // contact transient: largest (3 m/s²) for fast movements, ∝ (speed)²
  const contact = clamp(3 * ((SPEED_FACTOR.fast * st.duration_natural_s) / T) ** 2, 0.15, 3);

  return {
    hand: c.hand,
    orientation,
    duration_s: duration,
    p0,
    p1,
    grip,
    pose(tSinceOnset_s: number) {
      const t = clamp(tSinceOnset_s, 0, duration);
      const pr = progressAt(pieces, t);
      return path(pr.u, pr.du, pr.ddu);
    },
    path,
    h,
    nondominant,
    contact_ms2: contact,
  };
}

// ───────────────────────────── participants ─────────────────────────────

const ID_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function drawStyle(rng: () => number): SimStyle {
  // The net phone rotation (wrist + forearm) is drawn as a total angle split into pitch/yaw/roll,
  // so nobody gets an implausibly tiny rotation. Ranges are chosen so a natural-speed movement
  // peaks around 70–250 deg/s and 2–8 m/s², in line with hand-to-face movements measured with
  // wrist/hand IMUs.
  const duration = clamp(1.0 + 0.15 * gaussian(rng), 0.7, 1.35);
  const total = clamp(55 + 10 * gaussian(rng), 38, 90);
  const wp = Math.max(0.05, 0.65 + 0.12 * gaussian(rng));
  const wy = Math.max(0.05, 0.5 + 0.12 * gaussian(rng));
  const wr = Math.max(0.05, 0.35 + 0.12 * gaussian(rng));
  const wn = Math.sqrt(wp * wp + wy * wy + wr * wr);
  return {
    duration_natural_s: duration,
    yaw_net_deg: (total * wy) / wn,
    yaw_transient_deg: clamp(12 + 5 * gaussian(rng), 0, 28),
    roll_deg: (total * wr) / wn,
    pitch_deg: (total * wp) / wn,
    arc_m: clamp(0.05 + 0.02 * gaussian(rng), 0.01, 0.11),
    start_blend: 0.15 + 0.7 * rng(),
    reaction_ms: clamp(320 + 70 * gaussian(rng), 180, 650),
    tremor_dps: 0.9 + 1.1 * rng(),
    nondominant_slowdown: clamp(1.1 + 0.06 * gaussian(rng), 1, 1.35),
    nondominant_jitter: clamp(1.45 + 0.2 * gaussian(rng), 1.05, 2.2),
  };
}

function drawHeight(sex: Sex, z: number): number {
  const [mean, sd] = sex === 'male' ? [176, 7] : sex === 'female' ? [163, 6.5] : [170, 8.5];
  return Math.round(clamp(mean + sd * z, 145, 205));
}

export function randomSimParticipant(rng: () => number, overrides?: Partial<SimParticipant>): SimParticipant {
  const o = overrides ?? {};
  // fixed draw order; overrides never change how many numbers are consumed
  let id = 'P-';
  for (let i = 0; i < 6; i++) id += ID_ALPHABET[Math.floor(rng() * ID_ALPHABET.length) % ID_ALPHABET.length];
  const uSex = rng();
  const uAgeGroup = rng();
  const uAge = rng();
  const zHeight = gaussian(rng);
  const uHand = rng();
  const uCoin = rng();
  const zArm = gaussian(rng);
  const zShoulder = gaussian(rng);
  const style = drawStyle(rng);

  const sex: Sex = o.sex ?? (uSex < 0.49 ? 'female' : uSex < 0.98 ? 'male' : uSex < 0.99 ? 'other' : 'prefer_not_to_say');
  const age = o.age_years ?? (uAgeGroup < 0.75 ? 18 + Math.floor(uAge * 13) : 31 + Math.floor(uAge * 40));
  const height = o.height_cm ?? drawHeight(sex, zHeight);
  const handedness: HandednessSelf = o.handedness ?? (uHand < 0.88 ? 'right' : uHand < 0.98 ? 'left' : 'ambidextrous');
  const dominant: Hand = o.dominant ?? (handedness === 'ambidextrous' ? (uCoin < 0.5 ? 'right' : 'left') : handedness);
  const femaleScale = sex === 'female' ? 0.97 : 1;
  const out: SimParticipant = {
    participant_id: o.participant_id ?? id,
    sex,
    age_years: age,
    height_cm: height,
    handedness,
    dominant,
    arm_length_m: o.arm_length_m ?? (0.36 * height * (1 + 0.03 * clamp(zArm, -2.5, 2.5))) / 100,
    shoulder_half_width_m: o.shoulder_half_width_m ?? (0.105 * femaleScale * height * (1 + 0.05 * clamp(zShoulder, -2.5, 2.5))) / 100,
    style: o.style ?? style,
  };
  if (o.phone_hand !== undefined) out.phone_hand = o.phone_hand; // no draw: the stream stays as it was
  return out;
}

/** The hand a simulated participant picks when the choice is theirs: the usual phone hand, else the dominant one. */
export function simChoiceHand(p: SimParticipant): Hand {
  return p.phone_hand === 'left' || p.phone_hand === 'right' ? p.phone_hand : p.dominant;
}

/**
 * The condition a simulated participant actually performs for a planned one. Only street's 'choice'
 * trial (condition.hand_source 'reported') differs from the plan: its hand is a placeholder, so the
 * participant uses their own (simChoiceHand), and its 'free' grip is the natural one of the How-to
 * card — upright, screen toward them. Every other condition is returned unchanged (the same object).
 */
export function performedCondition(p: SimParticipant, c: TrialCondition): TrialCondition {
  if (c.hand_source !== 'reported') return c;
  return { ...c, hand: simChoiceHand(p), orientation: c.orientation === 'free' ? 'upright_toward' : c.orientation };
}

/**
 * Deterministic simulated participant for the in-app simulation mode: dominance from the profile's
 * handedness (ambidextrous → seeded coin flip), height → arm length (≈ 0.36 × height), sex/age and
 * the usual phone hand copied; style drawn from a mulberry32 stream seeded with `seed` (simRng =
 * util.makeRng's algorithm). profile null → a random right-hander. Missing age/height are drawn.
 */
export function simParticipantFromProfile(profile: ParticipantProfile | null, seed: number): SimParticipant {
  const rng = simRng(seed);
  if (!profile) return randomSimParticipant(rng, { handedness: 'right', dominant: 'right' });
  const overrides: Partial<SimParticipant> = {
    participant_id: profile.participant_id,
    sex: profile.sex,
    handedness: profile.handedness_self,
    phone_hand: profile.phone_hand,
  };
  if (profile.age_years !== null) overrides.age_years = profile.age_years;
  if (profile.height_cm !== null) overrides.height_cm = profile.height_cm;
  return randomSimParticipant(rng, overrides);
}

/**
 * Plausible platform parameters. Both platforms deliver devicemotion at 60 Hz (Chrome's pump and
 * WebKit's CoreMotion updates); Android has noisier timing and sensors and exposes the Generic
 * Sensor RelativeOrientationSensor (quaternion) plus the absolute (magnetometer-fused) orientation
 * streams, iOS Safari does not. Android (Chrome) values are quantised like Chromium's; iOS values are
 * full doubles.
 */
export function defaultSimPlatform(name: 'ios' | 'android', rng: () => number): SimPlatform {
  const ios = name === 'ios';
  const u = [rng(), rng(), rng(), rng(), rng()];
  const bias: Vec3 = [gaussian(rng), gaussian(rng), gaussian(rng)];
  const heading = 360 * rng();
  const biasSd = ios ? 0.08 : 0.25; // CoreMotion's deviceMotion is bias-compensated more aggressively
  return {
    name,
    gravitySign: ios ? IOS_GRAVITY_SIGN : 1,
    hz: 60,
    jitter_ms: ios ? 0.4 + 0.6 * u[0] : 1.2 + 1.8 * u[0],
    drop_prob: ios ? 0.001 + 0.003 * u[1] : 0.004 + 0.012 * u[1],
    acc_available: true,
    quaternion: !ios,
    acc_noise_ms2: ios ? 0.012 + 0.012 * u[2] : 0.018 + 0.025 * u[2],
    gyro_noise_dps: ios ? 0.12 + 0.12 * u[3] : 0.2 + 0.3 * u[3],
    gyro_bias_dps: [bias[0] * biasSd, bias[1] * biasSd, bias[2] * biasSd],
    heading_deg: heading,
    quantize: !ios, // no extra draw: platforms from the same rng stay identical
    absolute_orientation: !ios, // Chrome Android; Safari has neither stream (no extra draw either)
  };
}
