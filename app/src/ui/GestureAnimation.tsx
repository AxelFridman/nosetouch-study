/**
 * GestureAnimation: shows HOW to do the NoseTouch gesture as a looping illustration, readable in a few
 * seconds without text.
 *
 *  - SIDE view: a person in profile holds the phone with the arm straight out in front at shoulder
 *    height; the arm bends and brings the phone to the tip of the nose (a ring pulses at the touch),
 *    holds it there briefly and stretches out again. Shoulder, elbow and wrist are nested <g> groups
 *    whose origins are the joints; their angles come from two-link inverse kinematics along a smooth hand
 *    path with minimum-jerk timing. The person faces RIGHT for the right hand and LEFT for the left hand
 *    (the moving arm is the one nearest the viewer); the sleeve has the app's hand colour.
 *  - POV view ("what you see"): your own forearm enters from the bottom corner on the hand's side,
 *    holding the phone in the given grip. The phone is drawn with PhoneFigure's geometry, rotation
 *    semantics and "top" arrow (phoneFigureSpec: 200×200 box, body 76×140 centred on 100,100, clockwise
 *    rotation as the participant sees it); only the hand is mirrored for the left hand, never the phone.
 *    It starts small (far away, arm straight), grows as it comes in and only comes down onto the nose at
 *    the very end, on the hand's side of the nose, so the hand, the thumb and the sleeve stay in view
 *    during the whole touch and hold (tested for every grip, contact and variant).
 *  - Which part of the phone touches is NOT standardised: contact 'straight' (the default) shows the
 *    phone arriving upright (its face touches), 'angled' arriving tilted (its upper edge/corner touches),
 *    'any' alternates one loop of each.
 *  - hand=null (participant's choice): the loops alternate the right and the left hand (mirrored
 *    figure), in a neutral colour, with an outlined "?" on each panel.
 *
 * One loop of one hand with one contact is a "take" (GestureTimeline); a GesturePlan is the 1–2 takes
 * shown in turn. Animation = CSS keyframes only (transform + opacity), generated per plan from the pure
 * helpers below and injected in a <style> next to the drawing: no JS animation loop (the app samples
 * sensors at 60 Hz). All parts share the same duration and a zero delay, so a negative animation-delay
 * with animation-play-state: paused freezes the whole drawing at any time (gallery). Every animated
 * element ALSO carries its start pose as an SVG attribute: with playing=false (or if the <style> were
 * blocked) the drawing shows the start pose.
 * prefers-reduced-motion: still frames instead (arm straight → halfway → touching, plus the other
 * take's touch when the plan has two takes).
 * RTL: the wrapper is dir="ltr" and every coordinate is physical, so left stays left on Hebrew pages.
 * Colours come from CSS variables (gesture.css), light and dark.
 */
import { Fragment } from 'preact';
import { useMemo } from 'preact/hooks';
import { ORIENTATIONS } from '../core/constants';
import type { OrientationId } from '../core/types';
import { phoneFigureSpec, type FigureRotation, type FigureSide } from './PhoneFigure';
import './gesture.css';

// ───────────────────────────── public types ─────────────────────────────

export type GestureHand = 'left' | 'right' | null;
export type DrawnHand = 'left' | 'right';
export type GestureSpeed = 'fast' | 'natural' | 'slow';
/** which part of the phone touches: 'any' = one loop arriving upright, one arriving tilted */
export type GestureContact = 'straight' | 'angled' | 'any';
/** the contact shown in one take */
export type TakeContact = 'straight' | 'angled';
export type GestureVariant = 'standard' | 'curved' | 'pause';
export type GestureView = 'side' | 'pov' | 'both';
export type GripKind = 'portrait' | 'landscape' | 'flat';
export type GesturePhaseName = 'rest' | 'out' | 'pause' | 'hold' | 'back' | 'tail';

/** What the gesture is (everything that changes the motion). */
export interface GestureOptions {
  /** the hand to show; null = participant's choice (the loops alternate the right and the left hand) */
  hand: GestureHand;
  /** grip; default 'upright_toward'; unknown ids fall back to 'upright_toward' */
  orientation?: OrientationId;
  /** default 'natural' */
  speed?: GestureSpeed;
  /** default 'straight'; 'any' alternates a 'straight' and an 'angled' loop */
  contact?: GestureContact;
  /** default 'standard' */
  variant?: GestureVariant;
}

export interface GestureAnimationProps extends GestureOptions {
  /** default 'both' (side view and "what you see", side by side at every width) */
  view?: GestureView;
  /** default true; false shows a still of the start pose */
  playing?: boolean;
  /** accessible description (translated by the caller); English fallback */
  label?: string;
  /** at most 320 px wide: for a session card */
  compact?: boolean;
  /**
   * default true: the outlined "?" of the participant's choice (hand null). Cards that say "any hand"
   * in words pass false: on a phone the badge looked like a help button that does nothing.
   */
  badge?: boolean;
}

export interface GesturePhase {
  name: GesturePhaseName;
  /** seconds from the start of the take */
  start: number;
  end: number;
  /** path progress at the start / end of the phase (0 = arm straight, 1 = touching the nose) */
  from: number;
  to: number;
}

/** One take: one loop of one hand with one contact. */
export interface GestureTimeline {
  /** the hand drawn in this take */
  hand: DrawnHand;
  /** the participant chooses the hand (neutral sleeve colour, "?" badge) */
  choice: boolean;
  /** +1 right hand, −1 left hand */
  sign: 1 | -1;
  /** the side view shows the person facing this way (the moving arm is the one nearest the viewer) */
  faces: 'right' | 'left';
  /** POV: the corner the forearm enters from */
  povEntry: 'bottom-right' | 'bottom-left';
  orientation: OrientationId;
  /** = phoneFigureSpec(orientation).side */
  figureSide: FigureSide;
  /** = phoneFigureSpec(orientation).rotationDeg (clockwise, as the participant sees the phone) */
  rotationDeg: FigureRotation;
  grip: GripKind;
  speed: GestureSpeed;
  contact: TakeContact;
  variant: GestureVariant;
  /** pitch of the phone when it touches the nose, deg; + = upper edge tilted toward the face */
  contactTiltDeg: number;
  /** POV roll when it touches the nose, deg clockwise as the participant sees it (top leans inward) */
  contactRollDeg: number;
  /** one take, seconds */
  duration: number;
  phases: GesturePhase[];
  /** seconds: the movement starts / is halfway / touches the nose / leaves the nose / is back */
  moveStart: number;
  midAt: number;
  contactAt: number;
  holdEnd: number;
  returnEnd: number;
  /** the same instants as % of the take */
  pct: { moveStart: number; mid: number; contact: number; holdEnd: number; returnEnd: number };
  /** the stop of the 'pause' variant, seconds (0 otherwise) */
  pauseDuration: number;
}

/** The takes shown in turn (1 or 2) and the whole loop. */
export interface GesturePlan {
  /** stable id: CSS class + keyframe names */
  key: string;
  /** hand=null: the participant chooses */
  choice: boolean;
  /** the contact asked for ('any' = the takes alternate) */
  contact: GestureContact;
  takes: GestureTimeline[];
  /** start of each take in the loop, seconds */
  offsets: number[];
  /** the whole loop, seconds */
  duration: number;
}

// ───────────────────────────── timing ─────────────────────────────

/**
 * Phase durations in seconds. Takes: fast 1.6 s, natural 2.8 s, slow 5 s (+ the stop of 'pause', about
 * one second as the participants' instruction says). The movement TO the nose carries the speed; the
 * return is a little calmer when the approach is fast.
 */
export const SPEED_TIMING: Record<
  GestureSpeed,
  { rest: number; out: number; hold: number; back: number; tail: number; pause: number }
> = {
  fast: { rest: 0.3, out: 0.35, hold: 0.3, back: 0.5, tail: 0.15, pause: 0.8 },
  natural: { rest: 0.45, out: 0.85, hold: 0.45, back: 0.8, tail: 0.25, pause: 1.0 },
  slow: { rest: 0.5, out: 2.4, hold: 0.55, back: 1.35, tail: 0.2, pause: 1.0 },
};

/** 'angled': pitch of the phone at the touch (upper edge toward the face) and the POV roll (top inward,
 *  POV_TOUCH: counter-clockwise for the right hand). */
export const GESTURE_TILT_DEG = 32;
export const GESTURE_ROLL_DEG = 24;

const SPEEDS: GestureSpeed[] = ['fast', 'natural', 'slow'];
const CONTACTS: GestureContact[] = ['straight', 'angled', 'any'];
const VARIANTS: GestureVariant[] = ['standard', 'curved', 'pause'];

export function normalizeOrientation(o: unknown): OrientationId {
  return typeof o === 'string' && Object.prototype.hasOwnProperty.call(ORIENTATIONS, o)
    ? (o as OrientationId)
    : 'upright_toward';
}

function gripOf(side: FigureSide, rotationDeg: FigureRotation): GripKind {
  if (side === 'flat') return 'flat';
  return rotationDeg === 90 || rotationDeg === -90 ? 'landscape' : 'portrait';
}

const r6 = (x: number) => Math.round(x * 1e6) / 1e6;
const pctOf = (t: number, d: number) => r6((t / d) * 100);

interface NormalOptions {
  orientation: OrientationId;
  speed: GestureSpeed;
  contact: GestureContact;
  variant: GestureVariant;
}

function normalize(opts: GestureOptions): NormalOptions {
  return {
    orientation: normalizeOrientation(opts.orientation),
    speed: SPEEDS.includes(opts.speed as GestureSpeed) ? (opts.speed as GestureSpeed) : 'natural',
    contact: CONTACTS.includes(opts.contact as GestureContact) ? (opts.contact as GestureContact) : 'straight',
    variant: VARIANTS.includes(opts.variant as GestureVariant) ? (opts.variant as GestureVariant) : 'standard',
  };
}

function makeTake(o: NormalOptions, hand: DrawnHand, contact: TakeContact, choice: boolean): GestureTimeline {
  const spec = phoneFigureSpec(o.orientation);
  const sign: 1 | -1 = hand === 'left' ? -1 : 1;
  const T = SPEED_TIMING[o.speed];
  const phases: GesturePhase[] = [];
  let t = 0;
  const push = (name: GesturePhaseName, dur: number, from: number, to: number) => {
    phases.push({ name, start: t, end: r6(t + dur), from, to });
    t = r6(t + dur);
  };
  push('rest', T.rest, 0, 0);
  if (o.variant === 'pause') {
    push('out', T.out * 0.5, 0, 0.5);
    push('pause', T.pause, 0.5, 0.5);
    push('out', T.out * 0.5, 0.5, 1);
  } else {
    push('out', T.out, 0, 1);
  }
  push('hold', T.hold, 1, 1);
  push('back', T.back, 1, 0);
  push('tail', T.tail, 0, 0);

  const duration = t;
  const moveStart = phases[0].end;
  const hold = phases.find((p) => p.name === 'hold')!;
  const back = phases.find((p) => p.name === 'back')!;
  const pause = phases.find((p) => p.name === 'pause');
  const firstOut = phases.find((p) => p.name === 'out')!;
  // minimum-jerk is symmetric: half the distance at half the time
  const midAt = pause ? pause.start : r6((firstOut.start + firstOut.end) / 2);
  const angled = contact === 'angled';

  return {
    hand,
    choice,
    sign,
    faces: hand,
    povEntry: hand === 'left' ? 'bottom-left' : 'bottom-right',
    orientation: o.orientation,
    figureSide: spec.side,
    rotationDeg: spec.rotationDeg,
    grip: gripOf(spec.side, spec.rotationDeg),
    speed: o.speed,
    contact,
    variant: o.variant,
    contactTiltDeg: angled ? GESTURE_TILT_DEG : 0,
    // the top leans toward the middle of the face: counter-clockwise for the right hand (POV_TOUCH)
    contactRollDeg: angled ? sign * POV_TOUCH[gripOf(spec.side, spec.rotationDeg)].angled.roll : 0,
    duration,
    phases,
    moveStart,
    midAt,
    contactAt: hold.start,
    holdEnd: hold.end,
    returnEnd: back.end,
    pct: {
      moveStart: pctOf(moveStart, duration),
      mid: pctOf(midAt, duration),
      contact: pctOf(hold.start, duration),
      holdEnd: pctOf(hold.end, duration),
      returnEnd: pctOf(back.end, duration),
    },
    pauseDuration: pause ? r6(pause.end - pause.start) : 0,
  };
}

/**
 * The takes of a gesture, shown in turn (pure; unit-tested):
 *  - a hand + 'straight' or 'angled': one take;
 *  - a hand + 'any': [straight, angled] with that hand;
 *  - hand=null: the right hand, then the left one (with 'any': right + straight, then left + angled).
 */
export function gesturePlan(opts: GestureOptions): GesturePlan {
  const o = normalize(opts);
  const choice = !(opts.hand === 'left' || opts.hand === 'right');
  const hands: DrawnHand[] = choice ? ['right', 'left'] : [opts.hand as DrawnHand];
  const contacts: TakeContact[] = o.contact === 'any' ? ['straight', 'angled'] : [o.contact];
  const n = Math.max(hands.length, contacts.length);
  const takes: GestureTimeline[] = [];
  const offsets: number[] = [];
  let t = 0;
  for (let i = 0; i < n; i++) {
    const take = makeTake(o, hands[i % hands.length], contacts[i % contacts.length], choice);
    takes.push(take);
    offsets.push(t);
    t = r6(t + take.duration);
  }
  const who = choice ? 'x' : opts.hand === 'left' ? 'l' : 'r';
  return {
    key: `${who}-${o.orientation}-${o.speed}-${o.contact}-${o.variant}`,
    choice,
    contact: o.contact,
    takes,
    offsets,
    duration: t,
  };
}

/** The first take of the gesture (see gesturePlan for the whole loop). */
export function gestureTimeline(opts: GestureOptions): GestureTimeline {
  return gesturePlan(opts).takes[0];
}

/** A plan, or a single take shown on its own (older callers pass the timeline). */
export type PlanOrTake = GesturePlan | GestureTimeline;

/**
 * The plan of a single take. Its key names the take's own hand and contact, so it equals the key of the
 * same one-take gesture and never collides with a two-take plan (whose rules differ).
 */
export function planOf(x: PlanOrTake): GesturePlan {
  if ('takes' in x) return x;
  const who = (x.choice ? 'x' : '') + (x.hand === 'left' ? 'l' : 'r');
  return {
    key: `${who}-${x.orientation}-${x.speed}-${x.contact}-${x.variant}`,
    choice: x.choice,
    contact: x.contact,
    takes: [x],
    offsets: [0],
    duration: x.duration,
  };
}

/** Stable id of a gesture (CSS class + keyframe names). */
export function gestureKey(x: PlanOrTake): string {
  return planOf(x).key;
}

/** Does the plan switch hands between takes (participant's choice)? */
export function planSwitchesHands(plan: GesturePlan): boolean {
  return plan.takes.some((tk) => tk.hand !== plan.takes[0].hand);
}

// ───────────────────────────── small math ─────────────────────────────

export interface Pt {
  x: number;
  y: number;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const clamp01 = (x: number) => clamp(x, 0, 1);
const RAD = Math.PI / 180;
const toDeg = (r: number) => r / RAD;
/** minimum-jerk position profile (Flash & Hogan): smooth start and stop */
export const minJerk = (t: number) => {
  const u = clamp01(t);
  return u * u * u * (10 - 15 * u + 6 * u * u);
};
const smooth = (t: number) => {
  const u = clamp01(t);
  return u * u * (3 - 2 * u);
};
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const lerpPt = (a: Pt, b: Pt, u: number): Pt => ({ x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) });
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const bezier = (p0: Pt, p1: Pt, p2: Pt, u: number): Pt => {
  const v = 1 - u;
  return { x: v * v * p0.x + 2 * v * u * p1.x + u * u * p2.x, y: v * v * p0.y + 2 * v * u * p1.y + u * u * p2.y };
};
/** SVG rotation (degrees, clockwise on screen because y points down) */
const rotPt = (p: Pt, deg: number): Pt => {
  const c = Math.cos(deg * RAD);
  const s = Math.sin(deg * RAD);
  return { x: p.x * c - p.y * s, y: p.x * s + p.y * c };
};
const normDeg = (d: number) => {
  let x = d % 360;
  if (x > 180) x -= 360;
  if (x <= -180) x += 360;
  return x;
};
/** even-odd point-in-polygon test */
export function pointInPolygon(p: Pt, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** Path progress (0 = arm straight, 1 = touching the nose) at time t of a take (seconds, wraps around). */
export function progressAt(tl: GestureTimeline, t: number): number {
  const d = tl.duration;
  const x = ((t % d) + d) % d;
  for (const p of tl.phases) {
    if (x < p.end || p === tl.phases[tl.phases.length - 1]) {
      if (p.from === p.to) return p.from;
      const v = p.from + (p.to - p.from) * minJerk((x - p.start) / (p.end - p.start));
      return clamp(v, Math.min(p.from, p.to), Math.max(p.from, p.to));
    }
  }
  return 0;
}

// ───────────────────────────── side view model ─────────────────────────────

/** Side view: viewBox, joints and segment lengths of the RIGHT-facing figure (the left one is mirrored). */
export const SIDE_VIEW = { w: 190, h: 160 } as const;
export const SIDE_SHOULDER: Pt = { x: 44, y: 82 };
/** upper arm (shoulder → elbow) and forearm (elbow → wrist); the head is ~49 units tall */
export const UPPER_ARM = 65;
export const FOREARM = 54;
/** tip of the nose relative to the shoulder joint, and in the drawing */
const NOSE_REL: Pt = { x: 30, y: -37 };
export const SIDE_NOSE: Pt = { x: SIDE_SHOULDER.x + NOSE_REL.x, y: SIDE_SHOULDER.y + NOSE_REL.y };
/** the eye of the right-facing head */
export const SIDE_EYE: Pt = { x: SIDE_NOSE.x - 9.6, y: SIDE_NOSE.y - 11.6 };
/** the eye's radius */
export const SIDE_EYE_R = 1.6;
/**
 * The touch ring of the side view: centre (where the phone meets the tip of the nose), radius and stroke
 * widths at full size. It never grows beyond full size, so it always stays clear of the eye.
 */
export const SIDE_RING: Pt = { x: SIDE_NOSE.x + 1.5, y: SIDE_NOSE.y + 0.5 };
export const SIDE_RING_R = 4.2;
export const SIDE_RING_HALO_W = 3.8;
export const SIDE_RING_INK_W = 2;
const SIDE_MIRROR = `matrix(-1 0 0 1 ${SIDE_VIEW.w} 0)`;

/**
 * The phone as seen from the side, in the hand's frame (origin = wrist, phone upright when the hand's
 * absolute rotation is 0; the face is towards −x). Drawn ~1.4× life size so it reads at phone width;
 * the fist holds only its lower third. contact = where the nose touches when the phone arrives straight,
 * corner = the upper edge that touches when it arrives tilted.
 */
export interface SideGrip {
  bar: { x: number; y: number; w: number; h: number };
  contact: Pt;
  corner: Pt;
}
export const SIDE_GRIPS: Record<GripKind, SideGrip> = {
  portrait: { bar: { x: -11, y: -46, w: 7, h: 44 }, contact: { x: -11, y: -37 }, corner: { x: -11, y: -46 } },
  landscape: { bar: { x: -11, y: -37, w: 7, h: 23 }, contact: { x: -11, y: -30 }, corner: { x: -11, y: -37 } },
  flat: { bar: { x: -32, y: -24.5, w: 44, h: 7 }, contact: { x: -32, y: -21 }, corner: { x: -32, y: -24.5 } },
};
/** The fist around the lower part of the phone and the thumb over it, in the hand's frame. */
export const SIDE_FIST = { x: -9.5, y: -17.5, w: 18, h: 23.5, r: 6.5 } as const;
export const SIDE_THUMB: Capsule = { a: { x: 5.5, y: -12.5 }, b: { x: -7.5, y: -18.2 }, r: 3.3 };

export interface SidePose {
  /** SVG degrees (clockwise): upper arm from horizontal-forward, elbow and wrist relative to the parent */
  shoulder: number;
  elbow: number;
  wrist: number;
}

/** The point of the phone that touches the nose, in the hand's frame. */
const sideTouchLocal = (tl: GestureTimeline): Pt => {
  const g = SIDE_GRIPS[tl.grip];
  return tl.contact === 'angled' ? g.corner : g.contact;
};

/** Wrist path (relative to the shoulder): straight arm → at the nose; slight upward bulge. */
function sideTargets(tl: GestureTimeline): { w0: Pt; w1: Pt; wc: Pt } {
  const angled = tl.contact === 'angled';
  const touchLocal = rotPt(sideTouchLocal(tl), -tl.contactTiltDeg);
  // the touching point sits just in front of the nose tip (a hair above it when tilted)
  const touch: Pt = angled ? { x: NOSE_REL.x + 0.5, y: NOSE_REL.y - 0.6 } : { x: NOSE_REL.x + 0.8, y: NOSE_REL.y };
  const w1: Pt = { x: touch.x - touchLocal.x, y: touch.y - touchLocal.y };
  const w0: Pt = { x: UPPER_ARM + FOREARM, y: 0 };
  const wc: Pt = { x: (w0.x + w1.x) / 2, y: (w0.y + w1.y) / 2 - 8 };
  return { w0, w1, wc };
}

/** Two-link inverse kinematics, elbow below the shoulder→wrist line (SVG angles, degrees). */
function ik2(x: number, y: number): { a1: number; a2: number } {
  const L1 = UPPER_ARM;
  const L2 = FOREARM;
  const base = Math.atan2(y, x);
  let d = Math.hypot(x, y);
  if (d >= L1 + L2 - 1e-9) return { a1: toDeg(base), a2: 0 };
  d = Math.max(Math.abs(L1 - L2) + 1e-6, d);
  const alpha = Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
  const a1 = base + alpha;
  const ex = L1 * Math.cos(a1);
  const ey = L1 * Math.sin(a1);
  const af = Math.atan2(y - ey, x - ex);
  return { a1: toDeg(a1), a2: normDeg(toDeg(af - a1)) };
}

/** Joint angles of the side view at path progress s. */
export function sidePoseAt(tl: GestureTimeline, s: number): SidePose {
  const { w0, w1, wc } = sideTargets(tl);
  const u = clamp01(s);
  const w = bezier(w0, wc, w1, u);
  const { a1, a2 } = ik2(w.x, w.y);
  const hand = -tl.contactTiltDeg * smooth(u); // negative = top tilts toward the face (counter-clockwise)
  return { shoulder: a1, elbow: a2, wrist: normDeg(hand - a1 - a2) };
}

/** Forward kinematics of a side pose (absolute coordinates of the right-facing drawing). */
export function sideForward(tl: GestureTimeline, pose: SidePose): { elbow: Pt; wrist: Pt; touch: Pt; handDeg: number } {
  const S = SIDE_SHOULDER;
  const a1 = pose.shoulder;
  const a12 = pose.shoulder + pose.elbow;
  const handDeg = normDeg(a12 + pose.wrist);
  const elbow = { x: S.x + UPPER_ARM * Math.cos(a1 * RAD), y: S.y + UPPER_ARM * Math.sin(a1 * RAD) };
  const wrist = { x: elbow.x + FOREARM * Math.cos(a12 * RAD), y: elbow.y + FOREARM * Math.sin(a12 * RAD) };
  const local = rotPt(sideTouchLocal(tl), handDeg);
  return { elbow, wrist, touch: { x: wrist.x + local.x, y: wrist.y + local.y }, handDeg };
}

// ───────────────────────────── the dashed path (both views) ─────────────────────────────

/** arrow head length; the head ends this far in front of the nose tip */
export const TRAIL_HEAD = 5.5;
export const TRAIL_GAP = 4;

/**
 * Samples a moving point from s≈0 and stops where the arrow head (TRAIL_HEAD long, along the last
 * segment) ends TRAIL_GAP in front of the target; ~20 points. With `startOutside` (a polygon, e.g. the
 * phone where it starts), the path begins where the point has left that polygon.
 */
function trailTo(at: (s: number) => Pt, target: Pt, startOutside?: Pt[]): Pt[] {
  const stop = TRAIL_HEAD + TRAIL_GAP;
  const dense: Pt[] = [];
  let s0 = 0.02;
  if (startOutside) {
    while (s0 < 0.95 && pointInPolygon(at(s0), startOutside)) s0 += 0.005;
  }
  let prev = at(s0);
  dense.push(prev);
  for (let i = 1; i <= 400; i++) {
    const p = at(s0 + ((1 - s0) * i) / 400);
    const dp = dist(p, target);
    if (dp <= stop) {
      // exact crossing of the stop circle on the segment prev → p
      const dx = p.x - prev.x;
      const dy = p.y - prev.y;
      const fx = prev.x - target.x;
      const fy = prev.y - target.y;
      const a = dx * dx + dy * dy;
      const b = 2 * (fx * dx + fy * dy);
      const c = fx * fx + fy * fy - stop * stop;
      const disc = Math.max(0, b * b - 4 * a * c);
      const lam = a > 0 ? clamp01((-b - Math.sqrt(disc)) / (2 * a)) : 0;
      dense.push({ x: prev.x + lam * dx, y: prev.y + lam * dy });
      break;
    }
    dense.push(p);
    prev = p;
  }
  // keep ~20 points (always the first and the last two, which set the arrow head direction)
  const step = Math.max(1, Math.floor(dense.length / 20));
  const out = dense.filter((_, i) => i % step === 0 && i < dense.length - 2);
  out.push(dense[dense.length - 2] ?? dense[0], dense[dense.length - 1]);
  return out;
}

/** The arrow head (tip and base corners) at the end of a trail. */
export function trailHead(pts: Pt[]): Pt[] {
  const b = pts[pts.length - 1];
  const a = pts[pts.length - 2] ?? b;
  const ang = Math.atan2(b.y - a.y, b.x - a.x);
  return [
    { x: b.x + TRAIL_HEAD * Math.cos(ang), y: b.y + TRAIL_HEAD * Math.sin(ang) },
    { x: b.x + 4 * Math.cos(ang + 2.2), y: b.y + 4 * Math.sin(ang + 2.2) },
    { x: b.x + 4 * Math.cos(ang - 2.2), y: b.y + 4 * Math.sin(ang - 2.2) },
  ];
}

/** Side view: the path of the touching point (the face, or the upper corner when tilted) to the nose,
 *  in the drawing's coordinates (mirrored for the left hand). */
export function sideTrailPoints(tl: GestureTimeline): Pt[] {
  const pts = trailTo((s) => sideForward(tl, sidePoseAt(tl, s)).touch, SIDE_NOSE);
  return tl.faces === 'left' ? pts.map((p) => ({ x: SIDE_VIEW.w - p.x, y: p.y })) : pts;
}

/** the dashed path fades out over this part of the approach */
const trailGone = (tl: GestureTimeline) => tl.moveStart + 0.6 * (tl.contactAt - tl.moveStart);
const TRAIL_FADE_IN = 0.15;

/** Length of the fade-out / fade-in around the cut after take i when the next take uses the other hand. */
function cutFade(plan: GesturePlan, i: number): number {
  const a = plan.takes[i];
  const b = plan.takes[(i + 1) % plan.takes.length];
  return Math.min(0.15, 0.5 * (a.duration - a.returnEnd), 0.3 * b.moveStart);
}

/**
 * Opacity of the dashed path of take i at loop time t: shown while the arm rests before that take's
 * movement, fading out during the approach; it fades in during the final rest of the previous take
 * (never while the phone moves away), or right after the cut when the previous take used the other hand.
 */
export function trailOpacityAt(plan: GesturePlan, i: number, t: number): number {
  const D = plan.duration;
  const n = plan.takes.length;
  const x = ((t % D) + D) % D;
  const tk = plan.takes[i];
  const pi = (i - 1 + n) % n;
  const prev = plan.takes[pi];
  const rel = (((x - plan.offsets[i]) % D) + D) % D;
  const gone = trailGone(tk);
  if (rel <= tk.moveStart) {
    if (prev.hand !== tk.hand) {
      const a = cutFade(plan, pi);
      const b = a + TRAIL_FADE_IN;
      if (rel <= a) return 0;
      if (rel < b) return (rel - a) / (b - a);
    }
    return 1;
  }
  if (rel < gone) return 1 - (rel - tk.moveStart) / (gone - tk.moveStart);
  if (prev.hand === tk.hand) {
    const tailStart = plan.offsets[pi] + prev.returnEnd;
    const tailLen = prev.duration - prev.returnEnd;
    const relTail = (((x - tailStart) % D) + D) % D;
    if (relTail < tailLen) return relTail / tailLen;
  }
  return 0;
}

// ───────────────────────────── POV model ─────────────────────────────

/** POV: square viewBox; the tip of your own nose at the bottom centre. */
export const POV_VIEW = 200;
export const POV_NOSE: Pt = { x: 100, y: 138 };
/** apparent size of the far phone (arm straight); 1 = PhoneFigure's 200-unit box at full size */
export const POV_K0 = 0.29;
/** centre of the far phone (right hand): toward the hand's side and below eye level */
const POV_FAR: Pt = { x: 122, y: 82 };
/** where the forearm goes (toward its elbow, outside the frame): arm straight → bent at the touch */
const POV_ELBOW_FAR: Pt = { x: 240, y: 300 };
const POV_ELBOW_NEAR: Pt = { x: 215, y: 400 };
/** the forearm: skin from the wrist, then the sleeve (units of the phone box, scaled with the phone) */
export const POV_CUFF = 10;

/** A capsule (stadium): the segment a → b widened by r. */
export interface Capsule {
  a: Pt;
  b: Pt;
  r: number;
}
const cap = (ax: number, ay: number, bx: number, by: number, r: number): Capsule => ({
  a: { x: ax, y: ay },
  b: { x: bx, y: by },
  r,
});

/**
 * The RIGHT hand holding the phone, in PhoneFigure's box after the grip rotation (the left hand is its
 * mirror image in the box; the phone itself is never mirrored, so its grip reads exactly as in
 * PhoneFigure): the palm behind the phone, the fingertips in front of its far edge, the thumb in front of
 * its near side (thumb.b = the tip) and the wrist, where the forearm starts.
 */
export interface PovHand {
  palm: { c: Pt; rx: number; ry: number };
  fingers: Capsule[];
  thumb: Capsule;
  wrist: Pt;
}
export const POV_HANDS: Record<GripKind, PovHand> = {
  portrait: {
    palm: { c: { x: 140, y: 176 }, rx: 26, ry: 28 },
    fingers: [cap(51, 100, 68, 98, 6), cap(50, 113, 68, 112, 6), cap(51, 126, 67, 126, 6)],
    thumb: cap(154, 176, 134, 138, 7.5),
    wrist: { x: 146, y: 194 },
  },
  landscape: {
    palm: { c: { x: 172, y: 157 }, rx: 26, ry: 31 },
    fingers: [cap(128, 50, 128, 66, 6), cap(142, 49, 142, 66, 6), cap(156, 50, 156, 66, 6)],
    thumb: cap(180, 152, 160, 116, 7.5),
    wrist: { x: 178, y: 168 },
  },
  flat: {
    palm: { c: { x: 131, y: 161 }, rx: 31, ry: 21 },
    fingers: [cap(60, 126, 72, 124, 5.5), cap(66, 110, 78, 108, 5.5)],
    thumb: cap(150, 160, 146, 118, 6.5),
    wrist: { x: 132, y: 176 },
  },
};

/**
 * How the phone arrives at the nose, per grip and contact (right hand; mirrored for the left):
 *  - contact: the point of the phone (PhoneFigure's box after the grip rotation) that ends on the nose
 *    tip. straight = the inner, lower part of the face, so the phone hangs toward the hand and the hand
 *    and the sleeve stay in view; angled = the inner top corner (the body hangs toward the hand); flat =
 *    the near edge (a flat phone touches with its edge, not with its screen).
 *  - k1: apparent size at the touch (stylised: the hand and the sleeve must stay in view).
 *  - roll: in-plane rotation at the touch, degrees clockwise as the participant sees it.
 *  - py: vertical foreshortening at the touch (the phone tilted toward the face; 1 = facing you).
 *  - via: control point of the contact point's path, relative to the nose tip. The phone stays above
 *    (straight) or beside (angled) the nose and comes onto it only at the end of the movement.
 *  - swing ('curved'): how far the contact point first swings out to the hand's side (peak offset).
 */
export interface PovTouch {
  contact: Pt;
  k1: number;
  roll: number;
  py: number;
  via: Pt;
  swing: Pt;
}
export const POV_TOUCH: Record<GripKind, Record<TakeContact, PovTouch>> = {
  portrait: {
    straight: { contact: { x: 78, y: 146 }, k1: 0.72, roll: 0, py: 1, via: { x: 8, y: -46 }, swing: { x: 48, y: -14 } },
    angled: { contact: { x: 65.8, y: 33.8 }, k1: 0.6, roll: -GESTURE_ROLL_DEG, py: 0.6, via: { x: 40, y: -10 }, swing: { x: 14, y: -26 } },
  },
  landscape: {
    // the long phone: smaller when tilted, so the hand at its far end stays in view
    straight: { contact: { x: 70, y: 124 }, k1: 0.72, roll: 0, py: 1, via: { x: 8, y: -46 }, swing: { x: 26, y: -18 } },
    angled: { contact: { x: 33.8, y: 65.8 }, k1: 0.5, roll: -GESTURE_ROLL_DEG, py: 0.66, via: { x: 40, y: -10 }, swing: { x: 10, y: -24 } },
  },
  flat: {
    straight: { contact: { x: 76, y: 150 }, k1: 0.72, roll: 0, py: 1, via: { x: 8, y: -46 }, swing: { x: 48, y: -14 } },
    angled: { contact: { x: 58, y: 150 }, k1: 0.64, roll: -GESTURE_ROLL_DEG, py: 0.66, via: { x: 40, y: -10 }, swing: { x: 24, y: -18 } },
  },
};
/** 0 → 1 (at w = 1/3) → 0: the swing comes early, then the phone comes in to the nose */
const swingBump = (w: number) => {
  const u = clamp01(w);
  return 6.75 * u * (1 - u) * (1 - u);
};

/** Phone outline in PhoneFigure's 200×200 box (after the grip rotation). */
const POV_BODY: Record<GripKind, Pt[]> = {
  portrait: [
    { x: 62, y: 30 },
    { x: 138, y: 30 },
    { x: 138, y: 170 },
    { x: 62, y: 170 },
  ],
  landscape: [
    { x: 30, y: 62 },
    { x: 170, y: 62 },
    { x: 170, y: 138 },
    { x: 30, y: 138 },
  ],
  flat: [
    { x: 52, y: 150 },
    { x: 128, y: 150 },
    { x: 150, y: 80 },
    { x: 74, y: 80 },
  ],
};

/** PhoneFigure's "top" arrow of the upright phone: tip, head corners and shaft (the grip rotation turns it). */
const TOP_ARROW: Pt[] = [
  { x: 100, y: 2 },
  { x: 91, y: 15 },
  { x: 109, y: 15 },
  { x: 100, y: 21 },
  { x: 100, y: 27 },
];
/** PhoneFigure's arrow of the flat phone (the top edge points away from you). */
const FLAT_ARROW: Pt[] = [
  { x: 121.6, y: 49.6 },
  { x: 112.3, y: 59.6 },
  { x: 123.9, y: 63.2 },
  { x: 115.7, y: 68.3 },
  { x: 113.2, y: 76.2 },
];

/** The phone's "top" arrow (tip first) in PhoneFigure's box after the grip rotation; none for 'free'. */
export function povArrowBox(tl: GestureTimeline): Pt[] {
  if (tl.figureSide === 'free') return [];
  if (tl.figureSide === 'flat') return FLAT_ARROW;
  return TOP_ARROW.map((p) => {
    const v = rotPt({ x: p.x - 100, y: p.y - 100 }, tl.rotationDeg);
    return { x: 100 + v.x, y: 100 + v.y };
  });
}

export interface PovPose {
  /** centre of the phone box in the POV frame */
  x: number;
  y: number;
  /** apparent size (1 = PhoneFigure's 200-unit box at full size) */
  k: number;
  /** vertical foreshortening of the tilted phone (1 = facing you) */
  py: number;
  /** in-plane rotation, degrees clockwise */
  roll: number;
  /** the wrist (start of the forearm) in the frame */
  wx: number;
  wy: number;
  /** direction of the forearm from the wrist, degrees clockwise from +x */
  arm: number;
}

interface PovRaw {
  x: number;
  y: number;
  k: number;
  py: number;
  roll: number;
}
function povApply(p: Pt, pose: PovRaw): Pt {
  const v = rotPt({ x: (p.x - 100) * pose.k, y: (p.y - 100) * pose.k * pose.py }, pose.roll);
  return { x: pose.x + v.x, y: pose.y + v.y };
}
const mirrorBox = (p: Pt): Pt => ({ x: 200 - p.x, y: p.y });

/** Perspective-like progress: 1/size shrinks linearly with the distance, so most of the growth (and the
 *  descent onto the nose) happens at the end. */
const perspective = (u: number, k1: number) => {
  const rho = k1 / POV_K0;
  return u / (rho * (1 - u) + u);
};

/** POV pose at path progress s (the right hand computed, the left hand mirrored). */
export function povPoseAt(tl: GestureTimeline, s: number): PovPose {
  const g = POV_TOUCH[tl.grip][tl.contact];
  const u = clamp01(s);
  const w = perspective(u, g.k1);
  const k = POV_K0 + (g.k1 - POV_K0) * w;
  const e = smooth(u);
  const roll = g.roll * e;
  const py = 1 + (g.py - 1) * e;
  const c0: Pt = { x: POV_FAR.x + (g.contact.x - 100) * POV_K0, y: POV_FAR.y + (g.contact.y - 100) * POV_K0 };
  const via: Pt = { x: POV_NOSE.x + g.via.x, y: POV_NOSE.y + g.via.y };
  const base = bezier(c0, via, POV_NOSE, w);
  const b = tl.variant === 'curved' ? swingBump(w) : 0;
  const c: Pt = { x: base.x + g.swing.x * b, y: base.y + g.swing.y * b };
  const off = rotPt({ x: (g.contact.x - 100) * k, y: (g.contact.y - 100) * k * py }, roll);
  const raw: PovRaw = { x: c.x - off.x, y: c.y - off.y, k, py, roll };
  const wrist = povApply(POV_HANDS[tl.grip].wrist, raw);
  const elbow = lerpPt(POV_ELBOW_FAR, POV_ELBOW_NEAR, w);
  const arm = toDeg(Math.atan2(elbow.y - wrist.y, elbow.x - wrist.x));
  if (tl.sign === -1) {
    return { x: POV_VIEW - raw.x, y: raw.y, k, py, roll: -roll, wx: POV_VIEW - wrist.x, wy: wrist.y, arm: 180 - arm };
  }
  return { ...raw, wx: wrist.x, wy: wrist.y, arm };
}

/** Where a point of the (unmirrored) phone drawing lands in the POV frame. */
export function povPoint(pose: PovPose, p: Pt): Pt {
  return povApply(p, pose);
}
/** Where a point of the right hand's drawing lands (mirrored in the box for the left hand). */
export function povHandPoint(tl: GestureTimeline, pose: PovPose, p: Pt): Pt {
  return povApply(tl.sign === -1 ? mirrorBox(p) : p, pose);
}

/** The phone's outline in the POV frame at path progress s. */
export function povOutlineAt(tl: GestureTimeline, s: number): Pt[] {
  const pose = povPoseAt(tl, s);
  return POV_BODY[tl.grip].map((p) => povApply(p, pose));
}
/** The point of the phone that touches the nose, in the POV frame at path progress s. */
export function povContactAt(tl: GestureTimeline, s: number): Pt {
  return povHandPoint(tl, povPoseAt(tl, s), POV_TOUCH[tl.grip][tl.contact].contact);
}
/** The thumb tip in the POV frame. */
export function povThumbAt(tl: GestureTimeline, s: number): Pt {
  return povHandPoint(tl, povPoseAt(tl, s), POV_HANDS[tl.grip].thumb.b);
}
/** The phone's "top" arrow (tip first) in the POV frame; empty for 'free'. */
export function povArrowAt(tl: GestureTimeline, s: number): Pt[] {
  const pose = povPoseAt(tl, s);
  return povArrowBox(tl).map((p) => povApply(p, pose));
}
/** Unit direction of the forearm and the start of the sleeve (frame units). */
export function povSleeveAt(tl: GestureTimeline, s: number): { wrist: Pt; dir: Pt; cuff: Pt } {
  const p = povPoseAt(tl, s);
  const dir = { x: Math.cos(p.arm * RAD), y: Math.sin(p.arm * RAD) };
  return { wrist: { x: p.wx, y: p.wy }, dir, cuff: { x: p.wx + dir.x * POV_CUFF * p.k, y: p.wy + dir.y * POV_CUFF * p.k } };
}

/** Your own nose (POV), relative to its tip: cubic segments from the bottom-left round to the bottom-right. */
const NOSE_SEGMENTS: [Pt, Pt, Pt][] = [
  [{ x: -26, y: 56 }, { x: -15, y: 46 }, { x: -11.5, y: 33 }],
  [{ x: -9.5, y: 24 }, { x: -8.6, y: 13 }, { x: -6.2, y: 6.5 }],
  [{ x: -4, y: 0.5 }, { x: 4, y: 0.5 }, { x: 6.2, y: 6.5 }],
  [{ x: 8.6, y: 13 }, { x: 9.5, y: 24 }, { x: 11.5, y: 33 }],
  [{ x: 15, y: 46 }, { x: 26, y: 56 }, { x: 36, y: 66 }],
];
const NOSE_START: Pt = { x: -36, y: 66 };
const NOSE_D =
  `M${NOSE_START.x} ${NOSE_START.y}` + NOSE_SEGMENTS.map(([a, b, c]) => `C${a.x} ${a.y} ${b.x} ${b.y} ${c.x} ${c.y}`).join('') + 'Z';

/** The nose's silhouette in the POV frame (polygon, sampled from the drawn path). */
export function povNoseOutline(perSegment = 8): Pt[] {
  const out: Pt[] = [{ x: POV_NOSE.x + NOSE_START.x, y: POV_NOSE.y + NOSE_START.y }];
  let p0 = NOSE_START;
  for (const [a, b, c] of NOSE_SEGMENTS) {
    for (let i = 1; i <= perSegment; i++) {
      const t = i / perSegment;
      const v = 1 - t;
      const x = v * v * v * p0.x + 3 * v * v * t * a.x + 3 * v * t * t * b.x + t * t * t * c.x;
      const y = v * v * v * p0.y + 3 * v * v * t * a.y + 3 * v * t * t * b.y + t * t * t * c.y;
      out.push({ x: POV_NOSE.x + x, y: POV_NOSE.y + y });
    }
    p0 = c;
  }
  return out;
}

/** POV: the path of the touching point to the nose tip; it starts where it leaves the far phone. */
export function povTrailPoints(tl: GestureTimeline): Pt[] {
  return trailTo((s) => povContactAt(tl, s), POV_NOSE, povOutlineAt(tl, 0));
}

/** The touch burst: opacity and scale at time t of a take. */
export function burstAt(tl: GestureTimeline, t: number): { opacity: number; scale: number } {
  const d = tl.duration;
  const dt = (((t % d) + d) % d) - tl.contactAt;
  const fade = burstFade(tl);
  if (dt < 0 || dt > fade) return { opacity: 0, scale: 0.4 };
  if (dt < 0.08) return { opacity: dt / 0.08, scale: 0.4 + 0.3 * (dt / 0.08) };
  if (dt < 0.16) return { opacity: 1, scale: 0.7 + 0.3 * ((dt - 0.08) / 0.08) };
  const u = (dt - 0.16) / (fade - 0.16);
  return { opacity: 1 - u, scale: 1 + 0.25 * u };
}
const burstFade = (tl: GestureTimeline) => Math.min(0.42, tl.holdEnd - tl.contactAt + 0.12);

// ───────────────────────────── transforms + generated CSS ─────────────────────────────

const num = (x: number, d = 2) => {
  const v = Number(x.toFixed(d));
  return Object.is(v, -0) ? '0' : String(v);
};

type Fmt = 'css' | 'attr';
const tr = (f: Fmt, x: number, y: number) => (f === 'css' ? `translate(${num(x)}px,${num(y)}px)` : `translate(${num(x)} ${num(y)})`);
const ro = (f: Fmt, a: number) => (f === 'css' ? `rotate(${num(a)}deg)` : `rotate(${num(a)})`);
const sc = (f: Fmt, x: number, y = x) => (f === 'css' ? `scale(${num(x, 4)},${num(y, 4)})` : `scale(${num(x, 4)} ${num(y, 4)})`);

function sideTransforms(p: SidePose, f: Fmt) {
  return {
    sh: `${tr(f, SIDE_SHOULDER.x, SIDE_SHOULDER.y)} ${ro(f, p.shoulder)}`,
    el: `${tr(f, UPPER_ARM, 0)} ${ro(f, p.elbow)}`,
    wr: `${tr(f, FOREARM, 0)} ${ro(f, p.wrist)}`,
  };
}
function povTransforms(p: PovPose, f: Fmt) {
  return {
    pv: `${tr(f, p.x, p.y)} ${ro(f, p.roll)} ${sc(f, p.k, p.k * p.py)} ${tr(f, -100, -100)}`,
    pa: `${tr(f, p.wx, p.wy)} ${ro(f, p.arm)} ${sc(f, p.k)}`,
  };
}
/** the side ring's scale: as the burst's, but never beyond full size */
export const sideRingScale = (b: { scale: number }) => Math.min(1, b.scale);
const sideBurst = (b: { scale: number }, f: Fmt) => `${tr(f, SIDE_RING.x, SIDE_RING.y)} ${sc(f, sideRingScale(b))}`;
/** the POV burst: centred on the tip of your nose */
export const POV_BURST_CENTRE: Pt = { x: POV_NOSE.x, y: POV_NOSE.y - 1 };
const povBurst = (b: { scale: number }, f: Fmt) => `${tr(f, POV_BURST_CENTRE.x, POV_BURST_CENTRE.y)} ${sc(f, 1.45 * b.scale)}`;
const IDENTITY_CSS = 'matrix(1,0,0,1,0,0)';
const sideMirrorCss = (tl: GestureTimeline) => (tl.faces === 'left' ? `matrix(-1,0,0,1,${SIDE_VIEW.w},0)` : IDENTITY_CSS);
const handMirrorCss = (tl: GestureTimeline) => (tl.sign === -1 ? 'matrix(-1,0,0,1,200,0)' : IDENTITY_CSS);

/** Keyframe sample times of a take: every phase boundary, plus dense samples inside the movements. */
export function keyframeTimes(tl: GestureTimeline): number[] {
  const ts: number[] = [];
  for (const p of tl.phases) {
    const dur = p.end - p.start;
    const n = p.from === p.to ? 1 : clamp(Math.round(Math.abs(p.to - p.from) * 18 * Math.sqrt(dur / 0.85)), 6, 32);
    for (let i = 0; i <= n; i++) ts.push(r6(p.start + (dur * i) / n));
  }
  ts.sort((a, b) => a - b);
  return ts.filter((t, i) => i === 0 || t - ts[i - 1] > 1e-6);
}

type Frame = [number, string];

/** Concatenates per-take frames (times relative to each take) into one loop; a value that differs at a
 *  take boundary jumps there (the previous value is held until just before). */
function joinTakes(plan: GesturePlan, perTake: (tl: GestureTimeline, i: number) => Frame[]): Frame[] {
  const eps = plan.duration * 2e-5;
  const out: Frame[] = [];
  plan.takes.forEach((tl, i) => {
    for (const [t, v] of perTake(tl, i)) {
      const at = r6(plan.offsets[i] + t);
      const last = out[out.length - 1];
      if (last && at - last[0] < 1e-6) {
        if (last[1] === v) continue;
        last[0] = at - eps;
      }
      out.push([at, v]);
    }
  });
  return out;
}

/** Breakpoints of a piecewise-linear opacity over the loop, evaluated → frames. */
function opacityFrames(plan: GesturePlan, times: number[], at: (t: number) => number): Frame[] {
  const D = plan.duration;
  const ts = [0, D, ...times.map((t) => r6(((t % D) + D) % D))].sort((a, b) => a - b);
  const uniq = ts.filter((t, i) => i === 0 || t - ts[i - 1] > 1e-6);
  return uniq.map((t) => [t, `opacity:${num(at(t >= D ? 0 : t), 3)}`]);
}

/**
 * The @keyframes and animation rules of one gesture plan (pure). Selectors are scoped by the plan key,
 * so two instances of the same gesture share identical rules.
 */
export function gestureKeyframesCss(x: PlanOrTake, views: ('side' | 'pov')[] = ['side', 'pov']): string {
  const plan = planOf(x);
  const key = plan.key;
  const D = plan.duration;
  const pct = (t: number) => `${num((t / D) * 100, 4)}%`;
  const parts: { name: string; frames: Frame[] }[] = [];
  const switching = planSwitchesHands(plan);
  const times = plan.takes.map((tl) => keyframeTimes(tl));

  if (views.includes('side')) {
    const poses = plan.takes.map((tl, i) => times[i].map((t) => sideTransforms(sidePoseAt(tl, progressAt(tl, t)), 'css')));
    for (const name of ['sh', 'el', 'wr'] as const) {
      parts.push({ name, frames: joinTakes(plan, (_tl, i) => times[i].map((t, j) => [t, `transform:${poses[i][j][name]}`])) });
    }
    if (switching) {
      parts.push({ name: 'mi', frames: joinTakes(plan, (tl) => [[0, `transform:${sideMirrorCss(tl)}`], [tl.duration, `transform:${sideMirrorCss(tl)}`]]) });
    }
  }
  if (views.includes('pov')) {
    const poses = plan.takes.map((tl, i) => times[i].map((t) => povTransforms(povPoseAt(tl, progressAt(tl, t)), 'css')));
    for (const name of ['pv', 'pa'] as const) {
      parts.push({ name, frames: joinTakes(plan, (_tl, i) => times[i].map((t, j) => [t, `transform:${poses[i][j][name]}`])) });
    }
    if (switching) {
      parts.push({ name: 'hm', frames: joinTakes(plan, (tl) => [[0, `transform:${handMirrorCss(tl)}`], [tl.duration, `transform:${handMirrorCss(tl)}`]]) });
    }
  }
  const burstFrames = (fn: (b: { scale: number }, f: Fmt) => string) =>
    joinTakes(plan, (tl) => {
      const c = tl.contactAt;
      const d = tl.duration;
      const bt = [0, c, c + 0.08, c + 0.16, Math.min(d, c + burstFade(tl)), d].map(r6).filter((t, i, a) => i === 0 || t > a[i - 1]);
      return bt.map((t): Frame => {
        const b = t >= d ? burstAt(tl, 0) : burstAt(tl, t);
        return [t, `transform:${fn(b, 'css')};opacity:${num(b.opacity, 3)}`];
      });
    });
  if (views.includes('side')) parts.push({ name: 'sb', frames: burstFrames(sideBurst) });
  if (views.includes('pov')) parts.push({ name: 'pb', frames: burstFrames(povBurst) });

  // the dashed path of each take (shared by both views)
  plan.takes.forEach((tl, i) => {
    const n = plan.takes.length;
    const pi = (i - 1 + n) % n;
    const prev = plan.takes[pi];
    const o = plan.offsets[i];
    const a = cutFade(plan, pi);
    const bps = [o, o + tl.moveStart, o + trailGone(tl), plan.offsets[pi] + prev.returnEnd, o + a, o + a + TRAIL_FADE_IN];
    parts.push({ name: `tr${i}`, frames: opacityFrames(plan, bps, (t) => trailOpacityAt(plan, i, t)) });
  });

  // participant's choice: a short fade around the cut between the right- and the left-hand take
  if (switching) {
    parts.push({
      name: 'fd',
      frames: joinTakes(plan, (tl, i) => {
        const n = plan.takes.length;
        const pi = (i - 1 + n) % n;
        const fr: Frame[] = [];
        if (plan.takes[pi].hand !== tl.hand) fr.push([0, 'opacity:0'], [cutFade(plan, pi), 'opacity:1']);
        else fr.push([0, 'opacity:1']);
        const next = plan.takes[(i + 1) % n];
        if (next.hand !== tl.hand) fr.push([tl.duration - cutFade(plan, i), 'opacity:1'], [tl.duration, 'opacity:0']);
        else fr.push([tl.duration, 'opacity:1']);
        return fr;
      }),
    });
  }

  const dur = `${num(D, 3)}s`;
  return parts
    .map(
      (p) =>
        `@keyframes gx-${key}-${p.name}{${p.frames.map(([t, v]) => `${pct(t)}{${v}}`).join('')}}` +
        `.gx-k-${key} .gx-p-${p.name}{animation:gx-${key}-${p.name} ${dur} linear infinite}`,
    )
    .join('\n');
}

// ───────────────────────────── accessible description (English fallback) ─────────────────────────────

const GRIP_EN: Record<OrientationId, string> = {
  upright_toward: 'phone upright, screen facing you',
  upright_away: 'phone upright, screen facing away from you',
  top_left_toward: 'phone sideways with its top to your left, screen facing you',
  top_right_toward: 'phone sideways with its top to your right, screen facing you',
  upside_down_toward: 'phone upside down, screen facing you',
  upside_down_away: 'phone upside down, screen facing away from you',
  top_left_away: 'phone sideways with its top to your left, screen facing away from you',
  top_right_away: 'phone sideways with its top to your right, screen facing away from you',
  flat_up: 'phone flat, screen up',
  free: 'phone held the way you like',
};

/** English description used when the caller passes no (translated) label. */
export function gestureDescription(opts: GestureOptions): string {
  const plan = gesturePlan(opts);
  const tl = plan.takes[0];
  const who = plan.choice ? 'Either hand (shown with the right, then the left)' : tl.hand === 'left' ? 'Left hand' : 'Right hand';
  const pace = tl.speed === 'fast' ? 'quickly' : tl.speed === 'slow' ? 'slowly' : 'at a natural pace';
  const path = tl.variant === 'curved' ? ', swinging out to the side first' : tl.variant === 'pause' ? ', stopping halfway' : '';
  const touch =
    plan.contact === 'any'
      ? 'it touches the tip of your nose, once with the phone upright and once tilted so that its upper edge touches'
      : plan.contact === 'angled'
        ? 'it arrives tilted and its upper edge touches the tip of your nose'
        : 'it touches the tip of your nose';
  return (
    `Animation. ${who}, ${GRIP_EN[tl.orientation]}. Start with your arm straight out in front of you, ` +
    `bring the phone in ${pace}${path} until ${touch} (any part of the phone may touch), then stretch your arm out again.`
  );
}

// ───────────────────────────── drawing: shared ─────────────────────────────

/** capsule (stadium) between two points */
function capsule(x1: number, y1: number, x2: number, y2: number, r: number): string {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const nx = -Math.sin(a) * r;
  const ny = Math.cos(a) * r;
  const p = (x: number, y: number) => `${num(x)} ${num(y)}`;
  return (
    `M${p(x1 + nx, y1 + ny)}L${p(x2 + nx, y2 + ny)}A${r} ${r} 0 0 0 ${p(x2 - nx, y2 - ny)}` +
    `L${p(x1 - nx, y1 - ny)}A${r} ${r} 0 0 0 ${p(x1 + nx, y1 + ny)}Z`
  );
}

/** Dashed path of the phone, with an arrow head at its end (static; its group's opacity animates). */
function Trail(props: { pts: Pt[] }) {
  const pts = props.pts;
  if (pts.length < 3) return null;
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${num(p.x)} ${num(p.y)}`).join('');
  return (
    <g class="gx-trail-g">
      <path class="gx-trail" d={d} />
      <polygon class="gx-trail-head" points={trailHead(pts).map((p) => `${num(p.x)},${num(p.y)}`).join(' ')} />
    </g>
  );
}

/** Participant's choice: an outlined, neutral "?" (not a button). */
function ChoiceBadge(props: { x: number; y: number }) {
  return (
    <g class="gx-q-badge" transform={`translate(${props.x} ${props.y})`}>
      <circle r="8.5" />
      <text x="0" y="4.2" text-anchor="middle">
        ?
      </text>
    </g>
  );
}

// ───────────────────────────── drawing: side view ─────────────────────────────

/** Head in profile, facing right, relative to the tip of the nose (0, 0); about 50 units tall. */
const SIDE_HEAD =
  'M-23 -30.5C-14 -32 -7.5 -29 -5.6 -23C-4.4 -19 -2.6 -16.5 -3 -13.6C-3.4 -11.6 -5.4 -10.6 -5.6 -9.4' +
  'C-4.2 -6.2 -1.4 -3 0 -1.2C1 0.2 0.6 2.4 -1.4 2.9C-3 3.2 -4.8 2.8 -5.6 3.8C-5 5.2 -4 6.4 -4.4 7.6' +
  'C-4.8 8.6 -6 9 -6 9.6C-5 10.2 -4.6 11 -5 12C-5.4 13 -6.8 13.2 -6.7 14.4C-6.4 16 -5.2 17.2 -6.2 18.6' +
  'C-7.2 19.9 -9.4 20.1 -11 19.8C-17.3 19 -23.3 17 -27.8 13.2C-32.8 11.2 -37.3 10.2 -40.3 5.7' +
  'C-45.3 -1.8 -45.3 -13.8 -40.3 -21.3C-35.8 -27.8 -29.8 -31 -23 -30.5Z';

function SideBody() {
  const N = SIDE_NOSE;
  return (
    <g>
      {/* neck and torso (shirt), then the head on top */}
      <path class="gx-skin" d={`M${N.x - 38} ${N.y + 3}L${N.x - 41} ${N.y + 30}L${N.x - 16} ${N.y + 30}L${N.x - 13} ${N.y + 16}Z`} />
      <path
        class="gx-shirt"
        d="M22 172L19 104C18 86 24 72 34 70C42 68 50 69.5 56 71C61 74 64 86 64 102C64 122 61 140 61 172Z"
      />
      <path class="gx-line gx-thin" d="M35 70.5C43 76 51 75.5 56 71" />
      <g transform={`translate(${N.x} ${N.y})`}>
        <path class="gx-skin" d={SIDE_HEAD} />
        <path class="gx-line gx-thin" d="M-22.5 -12.5C-28 -14 -30 -3.5 -24 -1.5" />
        <path class="gx-line gx-thin" d="M-13 -16.2Q-9.6 -17.6 -6.4 -15.8" />
        <circle class="gx-eye" cx="-9.6" cy="-11.6" r="1.6" />
        <path class="gx-line gx-thin" d="M-6.2 9.7L-8.6 10.2" />
      </g>
    </g>
  );
}

function SidePhone(props: { tl: GestureTimeline }) {
  const { tl } = props;
  const b = SIDE_GRIPS[tl.grip].bar;
  if (tl.grip === 'flat') {
    // lying flat on the hand, screen up, top edge pointing away from the face
    return (
      <g>
        <rect class="gx-sph" x={b.x} y={b.y} width={b.w} height={b.h} rx="2" />
        <rect class="gx-ph-screen" x={b.x + 2.5} y={b.y + 1} width={b.w - 5} height="1.8" rx="0.9" />
      </g>
    );
  }
  const away = tl.figureSide === 'back';
  // the screen faces the face (−x) unless the grip says "screen away"; the camera bump sits on the back,
  // near the top edge (upright grips only: upside down it is inside the fist, sideways it faces the viewer)
  const screenX = away ? b.x + b.w - 2.6 : b.x + 0.8;
  const camX = away ? b.x - 1.6 : b.x + b.w;
  const showCam = tl.grip === 'portrait' && tl.rotationDeg === 0;
  return (
    <g>
      {showCam && <rect class="gx-sph" x={camX} y={b.y + 4} width="1.6" height="9" rx="0.7" />}
      <rect class="gx-sph" x={b.x} y={b.y} width={b.w} height={b.h} rx="2" />
      <rect class="gx-ph-screen" x={screenX} y={b.y + 2.2} width="1.8" height={b.h - 4.4} rx="0.9" />
    </g>
  );
}

function SideHand() {
  // a fist around the lower part of the phone (fingers curled on the face side), the thumb (near side)
  // lying over the top of the fist against the phone
  const f = SIDE_FIST;
  const t = SIDE_THUMB;
  return (
    <g>
      <rect class="gx-skin" x={f.x} y={f.y} width={f.w} height={f.h} rx={f.r} />
      <path class="gx-line gx-thin" d="M-9.5 -11.2H-5.2M-9.5 -5.6H-5M-9.5 0H-5.4" />
      <path class="gx-skin" d={capsule(t.a.x, t.a.y, t.b.x, t.b.y, t.r)} />
    </g>
  );
}

/** Side-view touch: a ring at the nose tip (clear of the eye; nothing spreads over the face). */
function TouchRing() {
  return (
    <g class="gx-burst">
      <circle class="gx-burst-halo" r={SIDE_RING_R} stroke-width={SIDE_RING_HALO_W} />
      <circle class="gx-burst-ink" r={SIDE_RING_R} stroke-width={SIDE_RING_INK_W} />
    </g>
  );
}

interface Still {
  take: number;
  s: number;
  touch: boolean;
}

function SideScene(props: { plan: GesturePlan; still?: Still; badge?: boolean }) {
  const { plan, still } = props;
  const live = !still;
  const tl = plan.takes[still?.take ?? 0];
  const pose = sidePoseAt(tl, still?.s ?? 0);
  const t = sideTransforms(pose, 'attr');
  const burst = still?.touch ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.4 };
  const switching = live && planSwitchesHands(plan);
  const cls = (name: string) => (live ? `gx-anim gx-p-${name}` : undefined);
  const trails = useMemo(() => (live ? plan.takes.map((tk) => sideTrailPoints(tk)) : []), [plan.key, live]);
  return (
    <svg class="gx-svg" viewBox={`0 0 ${SIDE_VIEW.w} ${SIDE_VIEW.h}`} aria-hidden="true" focusable="false">
      {trails.map((pts, i) => (
        <g key={i} class={cls(`tr${i}`)} opacity={i === 0 ? 1 : 0}>
          <Trail pts={pts} />
        </g>
      ))}
      <g class={switching ? 'gx-anim gx-p-fd' : undefined}>
        <g class={switching ? 'gx-anim gx-p-mi' : undefined} transform={tl.faces === 'left' ? SIDE_MIRROR : undefined}>
          <SideBody />
          <g class={cls('sh')} transform={t.sh}>
            {/* upper arm: sleeve in the hand colour */}
            <path class="gx-sleeve" d="M0 -9.5C20 -9.5 50 -8 65 -7A7 7 0 0 1 65 7C50 8 20 9.5 0 9.5A9.5 9.5 0 0 1 0 -9.5Z" />
            <g class={cls('el')} transform={t.el}>
              <path class="gx-sleeve" d="M0 -7C15 -6.8 34 -6 51 -5.4L51 5.4C34 6 15 6.8 0 7A7 7 0 0 1 0 -7Z" />
              <g class={cls('wr')} transform={t.wr}>
                <SidePhone tl={tl} />
                <SideHand />
              </g>
            </g>
          </g>
          <g class={cls('sb')} transform={sideBurst(burst, 'attr')} opacity={num(burst.opacity, 3)}>
            <TouchRing />
          </g>
        </g>
      </g>
      {live && plan.choice && props.badge !== false && <ChoiceBadge x={SIDE_VIEW.w - 13} y={13} />}
    </svg>
  );
}

// ───────────────────────────── drawing: POV ─────────────────────────────

// PhoneFigure's geometry (200×200 box): upright body 76×140 centred on (100, 100), same details.
const BODY = { x: 62, y: 30, w: 76, h: 140, r: 13 };
/** The 'free' grip's "?" (baseline centre; 60-unit glyph): in the upper half, clear of the nose at the touch. */
export const POV_Q: Pt = { x: 100, y: 100 };
export const POV_Q_SIZE = 60;

/** PhoneFigure's "top" arrow (drawn for the upright phone; the grip rotation turns it with the phone). */
function TopArrow() {
  return (
    <g class="gx-ph-arrow" aria-hidden="true">
      <line x1="100" y1="27" x2="100" y2="14" stroke-width="4" stroke-linecap="round" />
      <polygon points="100,2 91,15 109,15" />
    </g>
  );
}

function PovScreenSide(props: { question?: boolean }) {
  return (
    <g>
      <rect class="gx-ph" x={BODY.x} y={BODY.y} width={BODY.w} height={BODY.h} rx={BODY.r} />
      <rect class="gx-ph-screen" x={BODY.x + 5} y={BODY.y + 5} width={BODY.w - 10} height={BODY.h - 10} rx="9" />
      <rect class="gx-ph" x="88" y="39" width="24" height="7" rx="3.5" />
      <rect class="gx-ph" x="86" y="160" width="28" height="3" rx="1.5" />
      {props.question ? (
        <text class="gx-ph-q" x={POV_Q.x} y={POV_Q.y} text-anchor="middle">
          ?
        </text>
      ) : (
        // left-aligned "content" bars make the rotation readable, as in PhoneFigure
        <g class="gx-ph-content">
          <rect x="74" y="62" width="40" height="7" rx="3.5" />
          <rect x="74" y="78" width="52" height="5" rx="2.5" />
          <rect x="74" y="90" width="46" height="5" rx="2.5" />
          <rect x="74" y="102" width="50" height="5" rx="2.5" />
          <rect x="74" y="114" width="30" height="5" rx="2.5" />
        </g>
      )}
    </g>
  );
}

function PovBackSide() {
  return (
    <g>
      <rect class="gx-ph-back" x={BODY.x} y={BODY.y} width={BODY.w} height={BODY.h} rx={BODY.r} />
      <rect class="gx-ph-camera" x="68" y="36" width="31" height="31" rx="9" />
      <circle class="gx-ph-lens" cx="76.5" cy="44.5" r="5.2" />
      <circle class="gx-ph-lens" cx="76.5" cy="58.5" r="5.2" />
      <circle class="gx-ph-lens" cx="90.5" cy="51.5" r="5.2" />
      <circle class="gx-ph-flash" cx="91" cy="41" r="2.3" />
      <circle class="gx-ph-logo" cx="100" cy="104" r="9" />
    </g>
  );
}

const flatPt = (u: number, v: number) => `${num(52 + 76 * u + 22 * v, 1)},${num(150 - 70 * v, 1)}`;
function PovFlatPhone() {
  return (
    <g>
      <polygon class="gx-ph" points={[flatPt(0, 0), flatPt(1, 0), flatPt(1, 1), flatPt(0, 1)].join(' ')} stroke-linejoin="round" />
      <polygon class="gx-ph-screen" points={[flatPt(0.07, 0.06), flatPt(0.93, 0.06), flatPt(0.93, 0.94), flatPt(0.07, 0.94)].join(' ')} />
      <line class="gx-ph-notch" x1="101.8" y1="88.4" x2="117" y2="88.4" stroke-width="3.5" stroke-linecap="round" />
    </g>
  );
}

function PovPhone(props: { tl: GestureTimeline }) {
  const { tl } = props;
  if (tl.figureSide === 'flat') return <PovFlatPhone />;
  if (tl.figureSide === 'free') return <PovScreenSide question />;
  const rot = tl.rotationDeg;
  return (
    <g transform={rot === 0 ? undefined : `rotate(${rot} 100 100)`}>
      {tl.figureSide === 'screen' ? <PovScreenSide /> : <PovBackSide />}
    </g>
  );
}

/**
 * PhoneFigure's "top" arrow of the POV phone, drawn as its own layer with the phone's transform, above
 * your nose: it marks the grip, so the nose (in front of the phone) must never hide it.
 */
function PovArrow(props: { tl: GestureTimeline }) {
  const { tl } = props;
  if (tl.figureSide === 'free') return null;
  if (tl.figureSide === 'flat') {
    // PhoneFigure's flat arrow: the top edge points away from you
    return (
      <g class="gx-ph-arrow" aria-hidden="true">
        <line x1="113.2" y1="76.2" x2="118.2" y2="60.4" stroke-width="4" stroke-linecap="round" />
        <polygon points="121.6,49.6 112.3,59.6 123.9,63.2" />
      </g>
    );
  }
  const rot = tl.rotationDeg;
  return (
    <g transform={rot === 0 ? undefined : `rotate(${rot} 100 100)`}>
      <TopArrow />
    </g>
  );
}

/** The right hand behind the phone (palm), per grip (POV_HANDS). */
function PovHandBack(props: { grip: GripKind }) {
  const p = POV_HANDS[props.grip].palm;
  return <ellipse class="gx-skin gx-pov-stroke" cx={p.c.x} cy={p.c.y} rx={p.rx} ry={p.ry} />;
}
/** The right hand in front of the phone: fingertips on the far edge, the thumb on the near side. */
function PovHandFront(props: { grip: GripKind }) {
  const h = POV_HANDS[props.grip];
  return (
    <g class="gx-skin gx-pov-stroke">
      {[...h.fingers, h.thumb].map((c, i) => (
        <path key={i} d={capsule(c.a.x, c.a.y, c.b.x, c.b.y, c.r)} />
      ))}
    </g>
  );
}
function PovForearm() {
  // drawn along +x from the wrist (scaled with the phone); it widens toward you and leaves the frame
  return (
    <g>
      <path class="gx-skin gx-pov-stroke" d={`M0 -13L${POV_CUFF + 2} -14L${POV_CUFF + 2} 14L0 13Z`} />
      <path class="gx-sleeve gx-pov-stroke" d={`M${POV_CUFF} -14.5L640 -64L640 64L${POV_CUFF} 14.5Z`} />
    </g>
  );
}

function PovNose() {
  // your own nose as you see it: a rounded tip at the bottom centre of your view, the wings flaring out
  return (
    <g class="gx-nose" transform={`translate(${POV_NOSE.x} ${POV_NOSE.y})`}>
      <path d={NOSE_D} />
      <path class="gx-nose-shade" d="M-3.8 4.4C-5.6 10 -6.4 20 -8.2 30" />
    </g>
  );
}

function EyeIcon() {
  return (
    <g class="gx-eye-icon" transform="translate(9 9)">
      <path d="M2 9C6 3 11 1.5 15 1.5C19 1.5 24 3 28 9C24 15 19 16.5 15 16.5C11 16.5 6 15 2 9Z" />
      <circle cx="15" cy="9" r="4" />
    </g>
  );
}

/** POV touch: rays around the nose tip, outlined so they stand out on the screen and the nose. */
function Burst() {
  const angles = [-150, -90, -30, 30, 90, 150];
  const rays = angles.map((a) => {
    const c = Math.cos(a * RAD);
    const s = Math.sin(a * RAD);
    return { a, x1: num(5.6 * c), y1: num(5.6 * s), x2: num(10.4 * c), y2: num(10.4 * s) };
  });
  const layer = (cls: string, w: number) => (
    <g class={cls} stroke-width={w}>
      <circle r="3.4" />
      {rays.map((r) => (
        <line key={r.a} x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2} />
      ))}
    </g>
  );
  return (
    <g class="gx-burst">
      {layer('gx-burst-halo', 4.4)}
      {layer('gx-burst-ink', 2.4)}
    </g>
  );
}

function PovScene(props: { plan: GesturePlan; still?: Still; badge?: boolean }) {
  const { plan, still } = props;
  const live = !still;
  const tl = plan.takes[still?.take ?? 0];
  const pose = povPoseAt(tl, still?.s ?? 0);
  const t = povTransforms(pose, 'attr');
  const burst = still?.touch ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.4 };
  const switching = live && planSwitchesHands(plan);
  const cls = (name: string) => (live ? `gx-anim gx-p-${name}` : undefined);
  const mirror = tl.sign === -1 ? 'matrix(-1 0 0 1 200 0)' : undefined;
  const hm = switching ? 'gx-anim gx-p-hm' : undefined;
  const trails = useMemo(() => (live ? plan.takes.map((tk) => povTrailPoints(tk)) : []), [plan.key, live]);
  return (
    <svg class="gx-svg" viewBox={`0 0 ${POV_VIEW} ${POV_VIEW}`} aria-hidden="true" focusable="false">
      <g class={switching ? 'gx-anim gx-p-fd' : undefined}>
        <g class={cls('pa')} transform={t.pa}>
          <PovForearm />
        </g>
        <g class={cls('pv')} transform={t.pv}>
          <g class={hm} transform={mirror}>
            <PovHandBack grip={tl.grip} />
          </g>
          <PovPhone tl={tl} />
          <g class={hm} transform={mirror}>
            <PovHandFront grip={tl.grip} />
          </g>
        </g>
      </g>
      {/* the path of the phone, over the arm and the hand (it starts where it leaves the phone) */}
      {trails.map((pts, i) => (
        <g key={i} class={cls(`tr${i}`)} opacity={i === 0 ? 1 : 0}>
          <Trail pts={pts} />
        </g>
      ))}
      <PovNose />
      <g class={switching ? 'gx-anim gx-p-fd' : undefined}>
        <g class={cls('pv')} transform={t.pv}>
          <PovArrow tl={tl} />
        </g>
      </g>
      <g class={cls('pb')} transform={povBurst(burst, 'attr')} opacity={num(burst.opacity, 3)}>
        <Burst />
      </g>
      <EyeIcon />
      {live && plan.choice && props.badge !== false && <ChoiceBadge x={POV_VIEW - 15} y={15} />}
    </svg>
  );
}

// ───────────────────────────── component ─────────────────────────────

function FrameArrow() {
  return (
    <svg class="gx-farrow" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M2.5 8H12.5M8.5 4L12.5 8L8.5 12" />
    </svg>
  );
}
/** between the two alternative touches: "or" */
function FrameOr() {
  return (
    <svg class="gx-farrow gx-for" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M10.5 2.5L5.5 13.5" />
    </svg>
  );
}

/** Reduced motion: arm straight → halfway → touching (+ the other take's touch). */
export function stillFrames(plan: GesturePlan): Still[] {
  const frames: Still[] = [
    { take: 0, s: 0, touch: false },
    { take: 0, s: 0.5, touch: false },
    { take: 0, s: 1, touch: true },
  ];
  if (plan.takes.length > 1) frames.push({ take: 1, s: 1, touch: true });
  return frames;
}

function StaticRow(props: { plan: GesturePlan; view: 'side' | 'pov' }) {
  const { plan, view } = props;
  const frames = stillFrames(plan);
  return (
    <div class={`gx-frow gx-frow-${view}`}>
      {frames.map((f, i) => (
        <Fragment key={i}>
          {i > 0 && (i === 3 ? <FrameOr /> : <FrameArrow />)}
          {/* the step number goes UNDER the drawing: on it, it would cover the head or the touch at small sizes */}
          <div class="gx-frame">
            <div class={`gx-panel gx-panel-${view}`}>
              <div class={`gx-ratio gx-ratio-${view}`}>
                {view === 'side' ? <SideScene plan={plan} still={f} /> : <PovScene plan={plan} still={f} />}
              </div>
            </div>
            <span class="gx-fnum" aria-hidden="true">
              {Math.min(i + 1, 3)}
            </span>
          </div>
        </Fragment>
      ))}
    </div>
  );
}

export function GestureAnimation(props: GestureAnimationProps) {
  const view: GestureView = props.view === 'side' || props.view === 'pov' ? props.view : 'both';
  const playing = props.playing ?? true;
  const plan = useMemo(
    () =>
      gesturePlan({
        hand: props.hand,
        orientation: props.orientation,
        speed: props.speed,
        contact: props.contact,
        variant: props.variant,
      }),
    [props.hand, props.orientation, props.speed, props.contact, props.variant],
  );
  const key = plan.key;
  const views: ('side' | 'pov')[] = view === 'both' ? ['side', 'pov'] : [view];
  const css = useMemo(() => gestureKeyframesCss(plan, views), [key, view]);
  const label = props.label ?? gestureDescription(props);
  const first = plan.takes[0];
  const cls = [
    'gx',
    `gx-k-${key}`,
    `gx-hand-${plan.choice ? 'any' : first.hand}`,
    `gx-view-${view}`,
    props.compact ? 'gx-compact' : '',
    playing ? '' : 'gx-still',
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <div
      class={cls}
      dir="ltr"
      role="img"
      aria-label={label}
      data-gesture={key}
      data-takes={plan.takes.length}
      data-cycle-s={num(plan.duration, 3)}
      data-t-mid={num(first.midAt, 3)}
      data-t-touch={num(first.contactAt, 3)}
      data-t-touches={plan.takes.map((tk, i) => num(plan.offsets[i] + tk.contactAt, 3)).join(',')}
    >
      <style>{css}</style>
      <div class="gx-live">
        {views.includes('side') && (
          <div class="gx-panel gx-panel-side">
            <div class="gx-ratio gx-ratio-side">
              <SideScene plan={plan} badge={props.badge} />
            </div>
          </div>
        )}
        {views.includes('pov') && (
          <div class="gx-panel gx-panel-pov">
            <div class="gx-ratio gx-ratio-pov">
              <PovScene plan={plan} badge={props.badge} />
            </div>
          </div>
        )}
      </div>
      <div class="gx-frames">
        {views.map((v) => (
          <StaticRow key={v} plan={plan} view={v} />
        ))}
      </div>
    </div>
  );
}
