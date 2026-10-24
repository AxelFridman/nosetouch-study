/**
 * GestureAnimation (src/ui/GestureAnimation.tsx): the pure model behind the drawing. Each block pins one
 * property a participant relies on (which hand, which grip, that the touch is at the nose, ...); the
 * numbered items refer to the visual review of the first build.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { ALL_ORIENTATIONS } from '../src/core/constants';
import type { OrientationId } from '../src/core/types';
import {
  burstAt,
  GESTURE_TILT_DEG,
  gestureDescription,
  gestureKey,
  gestureKeyframesCss,
  gesturePlan,
  gestureTimeline,
  keyframeTimes,
  minJerk,
  planOf,
  planSwitchesHands,
  pointInPolygon,
  POV_BURST_CENTRE,
  POV_CUFF,
  POV_HANDS,
  POV_NOSE,
  POV_Q,
  POV_Q_SIZE,
  POV_TOUCH,
  POV_VIEW,
  povArrowAt,
  povArrowBox,
  povContactAt,
  povNoseOutline,
  povOutlineAt,
  povPoint,
  povPoseAt,
  povSleeveAt,
  povThumbAt,
  povTrailPoints,
  progressAt,
  SIDE_EYE,
  SIDE_EYE_R,
  SIDE_FIST,
  SIDE_GRIPS,
  SIDE_NOSE,
  SIDE_RING,
  SIDE_RING_HALO_W,
  SIDE_RING_R,
  SIDE_VIEW,
  sideForward,
  sidePoseAt,
  sideRingScale,
  sideTrailPoints,
  SPEED_TIMING,
  stillFrames,
  TRAIL_GAP,
  TRAIL_HEAD,
  trailHead,
  trailOpacityAt,
  type GestureContact,
  type GestureSpeed,
  type GestureTimeline,
  type GestureVariant,
  type Pt,
} from '../src/ui/GestureAnimation';
import { phoneFigureSpec } from '../src/ui/PhoneFigure';

const SPEEDS: GestureSpeed[] = ['fast', 'natural', 'slow'];
const HANDS = ['right', 'left'] as const;
const CONTACTS = ['straight', 'angled'] as const;
const VARIANTS: GestureVariant[] = ['standard', 'curved', 'pause'];
/** every grip the app can ask for (screen-away grips are never collected, but must still render well) */
const GRIPS = ALL_ORIENTATIONS;
const S_STEPS = Array.from({ length: 101 }, (_, i) => i / 100);

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
/** distance of a point inside the POV frame to its nearest edge (negative = outside) */
const frameMargin = (p: Pt) => Math.min(p.x, POV_VIEW - p.x, p.y, POV_VIEW - p.y);
const take = (o: Parameters<typeof gestureTimeline>[0]) => gestureTimeline(o);
/** all single takes with a drawn hand */
function* everyTake(): Generator<GestureTimeline> {
  for (const hand of HANDS)
    for (const orientation of GRIPS)
      for (const contact of CONTACTS) for (const variant of VARIANTS) yield take({ hand, orientation, contact, variant });
}
const label = (tl: GestureTimeline) => `${tl.hand}/${tl.orientation}/${tl.contact}/${tl.variant}`;

function distToSegment(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}
/** visible length of the sleeve: from the cuff along the forearm to the edge of the frame */
function sleeveInFrame(tl: GestureTimeline, s: number): number {
  const { cuff, dir } = povSleeveAt(tl, s);
  if (frameMargin(cuff) < 0) return 0;
  const ts: number[] = [];
  if (dir.x > 1e-9) ts.push((POV_VIEW - cuff.x) / dir.x);
  if (dir.x < -1e-9) ts.push(-cuff.x / dir.x);
  if (dir.y > 1e-9) ts.push((POV_VIEW - cuff.y) / dir.y);
  if (dir.y < -1e-9) ts.push(-cuff.y / dir.y);
  return Math.min(...ts);
}
function polygonArea(p: Pt[]): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const q = p[(i + 1) % p.length];
    a += p[i].x * q.y - q.x * p[i].y;
  }
  return Math.abs(a) / 2;
}
/** share of a convex polygon inside the POV frame (Sutherland–Hodgman clip) */
function shareInFrame(poly: Pt[]): number {
  const lerp = (a: Pt, b: Pt, t: number): Pt => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
  const clip = (pts: Pt[], inside: (p: Pt) => boolean, cut: (a: Pt, b: Pt) => Pt) => {
    const out: Pt[] = [];
    pts.forEach((a, i) => {
      const b = pts[(i + 1) % pts.length];
      if (inside(a)) out.push(a);
      if (inside(a) !== inside(b)) out.push(cut(a, b));
    });
    return out;
  };
  const W = POV_VIEW;
  let p = clip(poly, (q) => q.x >= 0, (a, b) => lerp(a, b, -a.x / (b.x - a.x)));
  p = clip(p, (q) => q.x <= W, (a, b) => lerp(a, b, (W - a.x) / (b.x - a.x)));
  p = clip(p, (q) => q.y >= 0, (a, b) => lerp(a, b, -a.y / (b.y - a.y)));
  p = clip(p, (q) => q.y <= W, (a, b) => lerp(a, b, (W - a.y) / (b.y - a.y)));
  return p.length < 3 ? 0 : polygonArea(p) / polygonArea(poly);
}
function segmentsCross(a: Pt, b: Pt, c: Pt, d: Pt): boolean {
  const o = (p: Pt, q: Pt, r: Pt) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
}
function polygonsOverlap(a: Pt[], b: Pt[]): boolean {
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) if (segmentsCross(a[i], a[(i + 1) % a.length], b[j], b[(j + 1) % b.length])) return true;
  return pointInPolygon(a[0], b) || pointInPolygon(b[0], a);
}
const centroid = (p: Pt[]): Pt => ({ x: p.reduce((s, q) => s + q.x, 0) / p.length, y: p.reduce((s, q) => s + q.y, 0) / p.length });
/** the hand's side of the drawing: +1 right, −1 left */
const sideOf = (tl: GestureTimeline) => (tl.hand === 'right' ? 1 : -1);

// ───────────────────────────── timing ─────────────────────────────

describe('timeline: durations and phases', () => {
  it('orders the loops fast < natural < slow, close to 1.6 s / 2.8 s / 5 s', () => {
    const d = SPEEDS.map((speed) => take({ hand: 'right', speed }).duration);
    expect(d[0]).toBeCloseTo(1.6, 6);
    expect(d[1]).toBeCloseTo(2.8, 6);
    expect(d[2]).toBeCloseTo(5.0, 6);
  });

  it('orders the movement to the nose itself fast < natural < slow', () => {
    const out = SPEEDS.map((speed) => {
      const tl = take({ hand: 'right', speed });
      return tl.contactAt - tl.moveStart;
    });
    expect(out[0]).toBeLessThan(out[1]);
    expect(out[1]).toBeLessThan(out[2]);
  });

  it('has contiguous phases rest, out, hold, back, tail and holds still at the nose', () => {
    const tl = take({ hand: 'left', speed: 'natural' });
    expect(tl.phases.map((p) => p.name)).toEqual(['rest', 'out', 'hold', 'back', 'tail']);
    expect(tl.phases[0].start).toBe(0);
    for (let i = 1; i < tl.phases.length; i++) expect(tl.phases[i].start).toBeCloseTo(tl.phases[i - 1].end, 9);
    expect(tl.phases[tl.phases.length - 1].end).toBeCloseTo(tl.duration, 9);
    expect(tl.holdEnd - tl.contactAt).toBeCloseTo(SPEED_TIMING.natural.hold, 9);
    for (const t of [tl.contactAt, (tl.contactAt + tl.holdEnd) / 2, tl.holdEnd - 1e-3]) expect(progressAt(tl, t)).toBe(1);
    expect(tl.pct.contact).toBeCloseTo((tl.contactAt / tl.duration) * 100, 6);
    expect(tl.pct.moveStart).toBeLessThan(tl.pct.mid);
    expect(tl.pct.mid).toBeLessThan(tl.pct.contact);
    expect(tl.pct.holdEnd).toBeLessThan(tl.pct.returnEnd);
  });

  it('pause stops halfway for about one second (as the instruction says) and lengthens the loop by exactly that', () => {
    for (const speed of SPEEDS) {
      const std = take({ hand: 'right', speed });
      const pau = take({ hand: 'right', speed, variant: 'pause' });
      const stop = pau.phases.find((p) => p.name === 'pause')!;
      expect(pau.pauseDuration).toBeGreaterThanOrEqual(0.8);
      expect(pau.pauseDuration).toBeLessThanOrEqual(1);
      expect(pau.duration).toBeCloseTo(std.duration + pau.pauseDuration, 6);
      expect(std.pauseDuration).toBe(0);
      expect(progressAt(pau, stop.start + 0.01)).toBe(0.5);
      expect(progressAt(pau, stop.end - 0.01)).toBe(0.5);
      expect(pau.midAt).toBeCloseTo(stop.start, 9);
    }
  });

  it('curved keeps the timing of the standard movement', () => {
    expect(take({ hand: 'right', variant: 'curved' }).duration).toBe(take({ hand: 'right' }).duration);
  });

  it('falls back to natural / straight / standard / upright for missing or unknown options', () => {
    const tl = take({ hand: 'right', speed: 'warp' as GestureSpeed, contact: undefined, variant: 'zigzag' as never });
    expect([tl.speed, tl.contact, tl.variant, tl.orientation]).toEqual(['natural', 'straight', 'standard', 'upright_toward']);
    expect(take({ hand: 'left', orientation: 'sideways_please' as OrientationId }).orientation).toBe('upright_toward');
    expect(gesturePlan({ hand: 'right', contact: 'diagonal' as GestureContact }).contact).toBe('straight');
  });

  it('minimum-jerk profile: 0 → 1, symmetric', () => {
    expect(minJerk(0)).toBe(0);
    expect(minJerk(1)).toBe(1);
    expect(minJerk(0.5)).toBeCloseTo(0.5, 12);
    expect(minJerk(0.25) + minJerk(0.75)).toBeCloseTo(1, 12);
  });
});

// ───────────────────────────── plans: hands and contacts shown in turn ─────────────────────────────

describe('plan: what the loop shows', () => {
  it('a hand with one contact is one take (the default contact is straight)', () => {
    for (const contact of [undefined, 'straight', 'angled'] as const) {
      const plan = gesturePlan({ hand: 'left', contact });
      expect(plan.takes).toHaveLength(1);
      expect(plan.takes[0].contact).toBe(contact ?? 'straight');
      expect(plan.takes[0].hand).toBe('left');
      expect(plan.duration).toBeCloseTo(plan.takes[0].duration, 9);
    }
  });

  it("'any' alternates an upright and a tilted touch with the same hand (item 9: not only the screen touches)", () => {
    const plan = gesturePlan({ hand: 'right', contact: 'any' });
    expect(plan.takes.map((t) => [t.hand, t.contact])).toEqual([
      ['right', 'straight'],
      ['right', 'angled'],
    ]);
    expect(plan.offsets).toEqual([0, plan.takes[0].duration]);
    expect(plan.duration).toBeCloseTo(plan.takes[0].duration + plan.takes[1].duration, 9);
    expect(planSwitchesHands(plan)).toBe(false);
  });

  it("participant's choice alternates the right and the left hand, both marked as a choice (item 10)", () => {
    const plan = gesturePlan({ hand: null });
    expect(plan.choice).toBe(true);
    expect(plan.takes.map((t) => [t.hand, t.faces, t.povEntry, t.choice])).toEqual([
      ['right', 'right', 'bottom-right', true],
      ['left', 'left', 'bottom-left', true],
    ]);
    expect(planSwitchesHands(plan)).toBe(true);
    const any = gesturePlan({ hand: null, contact: 'any' });
    expect(any.takes.map((t) => [t.hand, t.contact])).toEqual([
      ['right', 'straight'],
      ['left', 'angled'],
    ]);
  });

  it('gives every gesture its own stable key, the same for the plan and for its single take', () => {
    const keys = new Set<string>();
    for (const hand of ['right', 'left', null] as const)
      for (const contact of ['straight', 'angled', 'any'] as const)
        for (const variant of VARIANTS) keys.add(gesturePlan({ hand, contact, variant }).key);
    expect(keys.size).toBe(3 * 3 * 3);
    for (const k of keys) expect(k).toMatch(/^[a-z0-9_-]+$/);
    // older callers pass the timeline: the same key and the same CSS as the plan
    const plan = gesturePlan({ hand: 'left', orientation: 'top_left_toward', contact: 'angled' });
    expect(gestureKey(plan.takes[0])).toBe(gestureKey(plan));
    expect(gestureKeyframesCss(plan.takes[0])).toBe(gestureKeyframesCss(plan));
    expect(planOf(plan)).toBe(plan);
    // a lone take of a two-take loop gets its own key (its rules differ from the loop's)
    for (const opts of [{ hand: null }, { hand: 'right' as const, contact: 'any' as const }]) {
      const p2 = gesturePlan(opts);
      expect(new Set([gestureKey(p2), gestureKey(p2.takes[0]), gestureKey(p2.takes[1])]).size).toBe(3);
    }
  });
});

// ───────────────────────────── hand side ─────────────────────────────

describe('which hand: the physical side of the drawing', () => {
  it('maps the hand to the side the person faces and the corner the forearm enters from', () => {
    const r = take({ hand: 'right' });
    expect([r.sign, r.faces, r.povEntry]).toEqual([1, 'right', 'bottom-right']);
    const l = take({ hand: 'left' });
    expect([l.sign, l.faces, l.povEntry]).toEqual([-1, 'left', 'bottom-left']);
  });

  it('mirrors the POV exactly for the left hand; the phone itself is never mirrored (left stays physically left)', () => {
    for (const orientation of GRIPS)
      for (const contact of CONTACTS)
        for (const variant of VARIANTS) {
          const r = take({ hand: 'right', orientation, contact, variant });
          const l = take({ hand: 'left', orientation, contact, variant });
          for (const s of [0, 0.3, 0.5, 0.8, 1]) {
            const pr = povPoseAt(r, s);
            const pl = povPoseAt(l, s);
            expect(pl.x).toBeCloseTo(POV_VIEW - pr.x, 9);
            expect(pl.y).toBeCloseTo(pr.y, 9);
            expect(pl.k).toBeCloseTo(pr.k, 9);
            expect(pl.roll).toBeCloseTo(-pr.roll, 9);
            expect(pl.wx).toBeCloseTo(POV_VIEW - pr.wx, 9);
            expect(pl.arm).toBeCloseTo(180 - pr.arm, 9);
            const cr = povContactAt(r, s);
            const cl = povContactAt(l, s);
            expect(cl.x).toBeCloseTo(POV_VIEW - cr.x, 9);
            expect(cl.y).toBeCloseTo(cr.y, 9);
          }
        }
  });

  it('the forearm enters from the bottom corner on the hand side at every point of the movement', () => {
    for (const tl of everyTake()) {
      for (const s of S_STEPS) {
        const { dir, wrist } = povSleeveAt(tl, s);
        expect(dir.y, label(tl)).toBeGreaterThan(0.5); // towards the bottom edge
        expect(Math.sign(dir.x) * sideOf(tl), label(tl)).toBeGreaterThanOrEqual(0); // towards the hand's side
        expect((wrist.x - POV_NOSE.x) * sideOf(tl), label(tl)).toBeGreaterThan(10); // the hand is on its side
      }
    }
  });
});

// ───────────────────────────── item 1: the POV keeps the hand in view ─────────────────────────────

describe('POV: the hand stays in view for the whole movement, touch and hold (item 1)', () => {
  it('wrist, thumb and at least 15 units of sleeve stay inside the frame, for every grip, contact, variant and hand', () => {
    for (const tl of everyTake()) {
      let worstWrist = Infinity;
      let worstThumb = Infinity;
      let worstSleeve = Infinity;
      for (const s of S_STEPS) {
        worstWrist = Math.min(worstWrist, frameMargin(povSleeveAt(tl, s).wrist));
        worstThumb = Math.min(worstThumb, frameMargin(povThumbAt(tl, s)));
        worstSleeve = Math.min(worstSleeve, sleeveInFrame(tl, s));
      }
      expect(worstWrist, label(tl)).toBeGreaterThanOrEqual(5);
      expect(worstThumb, label(tl)).toBeGreaterThanOrEqual(5);
      expect(worstSleeve, label(tl)).toBeGreaterThanOrEqual(15);
    }
  });

  it('the phone stays at least 90% inside the frame at every point of the movement', () => {
    for (const tl of everyTake()) {
      const worst = Math.min(...S_STEPS.map((s) => shareInFrame(povOutlineAt(tl, s))));
      expect(worst, label(tl)).toBeGreaterThanOrEqual(0.9);
    }
  });

  it('the skin between the hand and the sleeve is short: the hand colour starts right at the wrist', () => {
    expect(POV_CUFF).toBeLessThanOrEqual(12);
  });
});

// ───────────────────────────── the touch is at the nose ─────────────────────────────

describe('the touch is at the tip of the nose', () => {
  it('POV: the touching point of the phone ends exactly on the nose tip, and the burst is centred there', () => {
    for (const tl of everyTake()) expect(dist(povContactAt(tl, 1), POV_NOSE), label(tl)).toBeLessThan(1e-6);
    expect(dist(POV_BURST_CENTRE, POV_NOSE)).toBeLessThanOrEqual(1.5);
  });

  it('side view: the touching point ends on the nose tip, for every grip and contact', () => {
    for (const o of GRIPS) {
      for (const contact of CONTACTS) {
        const tl = take({ hand: 'right', orientation: o, contact });
        expect(dist(sideForward(tl, sidePoseAt(tl, 1)).touch, SIDE_NOSE), `${o}/${contact}`).toBeLessThan(1.5);
      }
    }
  });

  it('side view: the touch ring sits on the nose tip and never reaches the eye (item 5)', () => {
    expect(dist(SIDE_RING, SIDE_NOSE)).toBeLessThanOrEqual(2);
    const tl = take({ hand: 'right', speed: 'slow' });
    let maxScale = 0;
    for (let t = 0; t < tl.duration; t += 0.01) maxScale = Math.max(maxScale, sideRingScale(burstAt(tl, t)));
    expect(maxScale).toBe(1);
    const outer = (SIDE_RING_R + SIDE_RING_HALO_W / 2) * maxScale;
    expect(dist(SIDE_RING, SIDE_EYE) - outer - SIDE_EYE_R).toBeGreaterThanOrEqual(8);
  });

  it('the burst flashes at the touch only', () => {
    for (const speed of SPEEDS) {
      const tl = take({ hand: 'right', speed });
      expect(burstAt(tl, 0).opacity).toBe(0);
      expect(burstAt(tl, tl.contactAt - 0.05).opacity).toBe(0);
      expect(burstAt(tl, tl.contactAt + 0.12).opacity).toBe(1);
      expect(burstAt(tl, tl.returnEnd).opacity).toBe(0);
    }
  });
});

// ───────────────────────────── item 2: the angled touch ─────────────────────────────

describe('angled touch (item 2)', () => {
  it('the side view arrives tilted 30–35° with the upper edge toward the face; straight arrives upright', () => {
    const angled = take({ hand: 'right', contact: 'angled' });
    expect(angled.contactTiltDeg).toBe(GESTURE_TILT_DEG);
    expect(GESTURE_TILT_DEG).toBeGreaterThanOrEqual(30);
    expect(GESTURE_TILT_DEG).toBeLessThanOrEqual(35);
    expect(sideForward(angled, sidePoseAt(angled, 1)).handDeg).toBeCloseTo(-GESTURE_TILT_DEG, 6);
    const straight = take({ hand: 'right' });
    expect(straight.contactTiltDeg).toBe(0);
    expect(sideForward(straight, sidePoseAt(straight, 1)).handDeg).toBeCloseTo(0, 6);
  });

  it('POV: the inner top corner touches and the phone hangs on the hand side of the nose', () => {
    for (const o of GRIPS) {
      for (const hand of HANDS) {
        const tl = take({ hand, orientation: o, contact: 'angled' });
        const outline = povOutlineAt(tl, 1);
        const c = centroid(outline);
        expect((c.x - POV_NOSE.x) * sideOf(tl), `${hand}/${o}`).toBeGreaterThan(10);
        if (tl.grip !== 'flat') {
          // the touching point is the phone's top corner nearest the middle of the face
          const touch = POV_TOUCH[tl.grip].angled.contact;
          const body = tl.grip === 'portrait' ? { x0: 62, y0: 30 } : { x0: 30, y0: 62 };
          expect(touch.x - body.x0).toBeLessThan(5);
          expect(touch.y - body.y0).toBeLessThan(5);
          // tilted: rolled toward the middle of the face and foreshortened
          const p = povPoseAt(tl, 1);
          expect(p.roll * sideOf(tl)).toBeLessThan(0);
          expect(p.py).toBeLessThan(1);
        }
      }
    }
    expect(povPoseAt(take({ hand: 'right' }), 1).roll).toBe(0);
    expect(povPoseAt(take({ hand: 'right' }), 1).py).toBe(1);
  });

  it('the roll the take reports is the roll drawn at the touch', () => {
    for (const tl of everyTake()) expect(povPoseAt(tl, 1).roll, label(tl)).toBeCloseTo(tl.contactRollDeg, 9);
  });
});

// ───────────────────────────── item 3: grips read as in PhoneFigure ─────────────────────────────

describe('grips: drawn exactly as PhoneFigure, top always visible (item 3)', () => {
  it('uses phoneFigureSpec for every orientation', () => {
    for (const o of GRIPS) {
      const tl = take({ hand: 'right', orientation: o });
      const spec = phoneFigureSpec(o);
      expect([tl.orientation, tl.rotationDeg, tl.figureSide]).toEqual([o, spec.rotationDeg, spec.side]);
    }
    expect(take({ hand: 'right', orientation: 'top_left_toward' }).grip).toBe('landscape');
    expect(take({ hand: 'right', orientation: 'top_right_away' }).grip).toBe('landscape');
    expect(take({ hand: 'right', orientation: 'flat_up' }).grip).toBe('flat');
    expect(take({ hand: 'right', orientation: 'upside_down_toward' }).grip).toBe('portrait');
    expect(take({ hand: 'right', orientation: 'free' }).grip).toBe('portrait');
  });

  it("the top arrow points where PhoneFigure's does, for both hands (top_left = the participant's left)", () => {
    const expected: Partial<Record<OrientationId, Pt>> = {
      upright_toward: { x: 0, y: -1 },
      upside_down_toward: { x: 0, y: 1 },
      top_left_toward: { x: -1, y: 0 },
      top_right_toward: { x: 1, y: 0 },
    };
    for (const [o, want] of Object.entries(expected) as [OrientationId, Pt][]) {
      for (const hand of HANDS) {
        for (const contact of CONTACTS) {
          const a = povArrowAt(take({ hand, orientation: o, contact }), 1);
          const tip = a[0];
          const tail = a[a.length - 1];
          const len = dist(tip, tail);
          const cos = ((tip.x - tail.x) * want.x + (tip.y - tail.y) * want.y) / len;
          // straight: exactly; angled: within the roll of the tilted phone
          expect(cos, `${hand}/${o}/${contact}`).toBeGreaterThan(contact === 'straight' ? 0.999 : Math.cos((30 * Math.PI) / 180));
        }
      }
    }
    expect(povArrowBox(take({ hand: 'right', orientation: 'free' }))).toEqual([]);
  });

  it('no finger or thumb covers the top arrow, whichever hand holds the phone', () => {
    for (const o of GRIPS) {
      for (const hand of HANDS) {
        const tl = take({ hand, orientation: o });
        const h = POV_HANDS[tl.grip];
        const mirror = (p: Pt): Pt => (hand === 'left' ? { x: 200 - p.x, y: p.y } : p);
        for (const a of povArrowBox(tl)) {
          for (const c of [...h.fingers, h.thumb]) {
            expect(distToSegment(a, mirror(c.a), mirror(c.b)) - c.r, `${hand}/${o}`).toBeGreaterThan(3);
          }
        }
      }
    }
  });

  it('the top arrow stays inside the frame at every point of the movement', () => {
    // (it is drawn above your nose, so the nose never hides it)
    for (const tl of everyTake()) {
      if (tl.orientation === 'free') continue;
      const worst = Math.min(...S_STEPS.flatMap((s) => povArrowAt(tl, s).map(frameMargin)));
      expect(worst, label(tl)).toBeGreaterThanOrEqual(3);
    }
  });

  it('flat: the near edge touches, not the screen (item 18); free: the "?" stays clear of the nose (item 19)', () => {
    for (const contact of CONTACTS) {
      const c = POV_TOUCH.flat[contact].contact;
      expect(c.y).toBe(150); // the near edge of PhoneFigure's flat phone runs from (52, 150) to (128, 150)
      expect(c.x).toBeGreaterThan(52);
      expect(c.x).toBeLessThan(128);
    }
    // the glyph's box: from its baseline up to ~75% of the font size, ±30% of it to each side
    const w = 0.3 * POV_Q_SIZE;
    const h = 0.75 * POV_Q_SIZE;
    const glyph: Pt[] = [
      { x: POV_Q.x - w, y: POV_Q.y },
      { x: POV_Q.x + w, y: POV_Q.y },
      { x: POV_Q.x + w, y: POV_Q.y - h },
      { x: POV_Q.x - w, y: POV_Q.y - h },
    ];
    const nose = povNoseOutline();
    for (const hand of HANDS) {
      for (const contact of CONTACTS) {
        const tl = take({ hand, orientation: 'free', contact });
        for (const s of S_STEPS) {
          const pose = povPoseAt(tl, s);
          expect(polygonsOverlap(glyph.map((p) => povPoint(pose, p)), nose), `${hand}/${contact} s=${s}`).toBe(false);
        }
      }
    }
  });
});

// ───────────────────────────── item 6: no contact before the touch ─────────────────────────────

describe('POV: the phone reaches the nose only at the end (item 6)', () => {
  it('the phone does not overlap the nose before 85% of the path', () => {
    const nose = povNoseOutline();
    for (const tl of everyTake()) {
      for (const s of S_STEPS.filter((x) => x <= 0.85)) {
        expect(polygonsOverlap(povOutlineAt(tl, s), nose), `${label(tl)} at s=${s}`).toBe(false);
      }
    }
  });

  it('at the halfway instant the whole phone is still above the nose tip (straight)', () => {
    for (const o of GRIPS) {
      const tl = take({ hand: 'right', orientation: o });
      const lowest = Math.max(...povOutlineAt(tl, progressAt(tl, tl.midAt)).map((p) => p.y));
      expect(lowest, o).toBeLessThan(POV_NOSE.y - 3);
    }
  });

  it('the phone starts small and far, then looms up to the nose', () => {
    for (const tl of everyTake()) {
      const ks = [0, 0.25, 0.5, 0.75, 1].map((s) => povPoseAt(tl, s).k);
      for (let i = 1; i < ks.length; i++) expect(ks[i], label(tl)).toBeGreaterThan(ks[i - 1]);
      expect(ks[4] / ks[0], label(tl)).toBeGreaterThan(1.6);
    }
  });
});

// ───────────────────────────── items 4, 15, 16: the dashed path ─────────────────────────────

describe('the dashed path points at the nose (items 4, 15, 16)', () => {
  it('side view: the arrow head ends just in front of the nose tip and points at it, never at the eye', () => {
    for (const o of GRIPS) {
      for (const contact of CONTACTS) {
        for (const hand of HANDS) {
          const tl = take({ hand, orientation: o, contact });
          const pts = sideTrailPoints(tl);
          const [tip] = trailHead(pts);
          const nose = hand === 'right' ? SIDE_NOSE : { x: SIDE_VIEW.w - SIDE_NOSE.x, y: SIDE_NOSE.y };
          const eye = hand === 'right' ? SIDE_EYE : { x: SIDE_VIEW.w - SIDE_EYE.x, y: SIDE_EYE.y };
          const id = `${hand}/${o}/${contact}`;
          expect(dist(tip, nose), id).toBeGreaterThanOrEqual(TRAIL_GAP - 0.5);
          expect(dist(tip, nose), id).toBeLessThanOrEqual(TRAIL_GAP + 1);
          expect(dist(tip, eye), id).toBeGreaterThan(dist(tip, nose) + 6);
          // the head's direction: toward the nose tip
          const base = pts[pts.length - 1];
          const dir = { x: tip.x - base.x, y: tip.y - base.y };
          const to = { x: nose.x - tip.x, y: nose.y - tip.y };
          const cos = (dir.x * to.x + dir.y * to.y) / (Math.hypot(dir.x, dir.y) * Math.hypot(to.x, to.y));
          expect(cos, id).toBeGreaterThan(Math.cos((25 * Math.PI) / 180));
        }
      }
    }
  });

  it('POV: the path starts where it leaves the phone and its head ends just above the nose tip', () => {
    for (const tl of everyTake()) {
      const pts = povTrailPoints(tl);
      expect(pointInPolygon(pts[0], povOutlineAt(tl, 0)), label(tl)).toBe(false);
      const [tip] = trailHead(pts);
      expect(dist(tip, POV_NOSE), label(tl)).toBeGreaterThanOrEqual(TRAIL_GAP - 0.5);
      expect(dist(tip, POV_NOSE), label(tl)).toBeLessThanOrEqual(TRAIL_GAP + 1);
      expect(tip.y, label(tl)).toBeLessThan(POV_NOSE.y); // it arrives from above or beside, not through the nose
      expect(TRAIL_HEAD).toBeGreaterThan(3);
    }
  });

  it('the path shows before the movement and fades in again only after the phone is back (item 15)', () => {
    for (const opts of [
      { hand: 'right' as const },
      { hand: 'left' as const, contact: 'any' as const },
      { hand: null, contact: 'any' as const, variant: 'pause' as const },
    ]) {
      const plan = gesturePlan(opts);
      plan.takes.forEach((tl, i) => {
        const o = plan.offsets[i];
        for (let t = o + tl.moveStart + 0.6 * (tl.contactAt - tl.moveStart) + 1e-3; t < o + tl.returnEnd; t += 0.01) {
          expect(trailOpacityAt(plan, i, t), `${plan.key} take ${i} t=${t.toFixed(2)}`).toBe(0);
        }
        // just before its own movement it is fully shown
        expect(trailOpacityAt(plan, i, o + tl.moveStart - 0.01)).toBe(1);
        // the other take's path never shows during this take's movement
        plan.takes.forEach((_, j) => {
          if (j !== i) for (let t = o + tl.moveStart; t < o + tl.returnEnd; t += 0.02) expect(trailOpacityAt(plan, j, t)).toBe(0);
        });
      });
    }
  });

  it('curved: the path first swings clearly out to the hand side (item 16)', () => {
    for (const hand of HANDS) {
      for (const o of ['upright_toward', 'top_left_toward', 'top_right_toward', 'flat_up'] as OrientationId[]) {
        const std = take({ hand, orientation: o });
        const cur = take({ hand, orientation: o, variant: 'curved' });
        let out = 0;
        for (const s of S_STEPS) out = Math.max(out, (povContactAt(cur, s).x - povContactAt(std, s).x) * sideOf(std));
        expect(out, `${hand}/${o}`).toBeGreaterThan(15);
        expect(dist(povContactAt(cur, 1), povContactAt(std, 1))).toBeLessThan(1e-9);
      }
    }
  });
});

// ───────────────────────────── side view ─────────────────────────────

describe('side view', () => {
  it('starts with the arm fully straight and horizontal, the phone upright', () => {
    for (const contact of CONTACTS) {
      const p = sidePoseAt(take({ hand: 'right', contact }), 0);
      expect(p.shoulder).toBeCloseTo(0, 6);
      expect(p.elbow).toBeCloseTo(0, 6);
      expect(p.wrist).toBeCloseTo(0, 6);
    }
  });

  it('bends the elbow a natural amount (flexion 110–150°) with the elbow below the shoulder', () => {
    for (const contact of CONTACTS) {
      const tl = take({ hand: 'right', contact });
      const p = sidePoseAt(tl, 1);
      expect(-p.elbow).toBeGreaterThan(110);
      expect(-p.elbow).toBeLessThan(150);
      expect(p.shoulder).toBeGreaterThan(20);
      expect(sideForward(tl, p).elbow.y).toBeGreaterThan(sideForward(tl, sidePoseAt(tl, 0)).elbow.y);
    }
  });

  it('the phone is large enough to read and mostly outside the fist (item 11)', () => {
    const bar = SIDE_GRIPS.portrait.bar;
    expect(bar.h).toBeGreaterThanOrEqual(40);
    const visible = SIDE_FIST.y - bar.y; // from the top of the phone down to the top of the fist
    expect(visible / bar.h).toBeGreaterThanOrEqual(0.6);
  });

  it('has keyframes dense enough: CSS linear interpolation stays within 1.5 units of the true path', () => {
    for (const speed of SPEEDS) {
      for (const variant of ['standard', 'pause'] as const) {
        for (const contact of CONTACTS) {
          const tl = take({ hand: 'right', speed, variant, contact });
          const ts = keyframeTimes(tl);
          let worst = 0;
          for (let i = 1; i < ts.length; i++) {
            const a = sidePoseAt(tl, progressAt(tl, ts[i - 1]));
            const b = sidePoseAt(tl, progressAt(tl, ts[i]));
            const lin = { shoulder: (a.shoulder + b.shoulder) / 2, elbow: (a.elbow + b.elbow) / 2, wrist: (a.wrist + b.wrist) / 2 };
            const truth = sidePoseAt(tl, progressAt(tl, (ts[i - 1] + ts[i]) / 2));
            const fl = sideForward(tl, lin);
            const ft = sideForward(tl, truth);
            worst = Math.max(worst, dist(fl.touch, ft.touch), dist(fl.wrist, ft.wrist));
          }
          expect(worst, `${speed}/${variant}/${contact}`).toBeLessThan(1.5);
        }
      }
    }
  });

  it('POV keyframes are dense enough too: the phone centre stays within 2 units of the true path', () => {
    for (const tl of everyTake()) {
      if (tl.speed !== 'natural') continue;
      const ts = keyframeTimes(tl);
      let worst = 0;
      for (let i = 1; i < ts.length; i++) {
        const a = povPoseAt(tl, progressAt(tl, ts[i - 1]));
        const b = povPoseAt(tl, progressAt(tl, ts[i]));
        const m = povPoseAt(tl, progressAt(tl, (ts[i - 1] + ts[i]) / 2));
        worst = Math.max(worst, Math.hypot((a.x + b.x) / 2 - m.x, (a.y + b.y) / 2 - m.y));
      }
      expect(worst, label(tl)).toBeLessThan(2);
    }
  });
});

// ───────────────────────────── generated CSS ─────────────────────────────

describe('generated keyframes', () => {
  it('samples every phase boundary and stays within the take', () => {
    const tl = take({ hand: 'left', speed: 'slow', variant: 'pause' });
    const ts = keyframeTimes(tl);
    expect(ts[0]).toBe(0);
    expect(ts[ts.length - 1]).toBeCloseTo(tl.duration, 9);
    for (const p of tl.phases) expect(ts.some((t) => Math.abs(t - p.start) < 1e-6)).toBe(true);
    for (let i = 1; i < ts.length; i++) expect(ts[i]).toBeGreaterThan(ts[i - 1]);
  });

  it('scopes the rules by the gesture key, with increasing percentages and finite numbers', () => {
    const plan = gesturePlan({ hand: 'right', orientation: 'top_left_toward', contact: 'angled', variant: 'curved' });
    const key = gestureKey(plan);
    const css = gestureKeyframesCss(plan);
    expect(css).not.toMatch(/NaN|Infinity|undefined/);
    for (const part of ['sh', 'el', 'wr', 'sb', 'pv', 'pa', 'pb', 'tr0']) {
      expect(css).toContain(`@keyframes gx-${key}-${part}{`);
      expect(css).toContain(`.gx-k-${key} .gx-p-${part}{animation:gx-${key}-${part} 2.8s linear infinite}`);
    }
    for (const m of css.matchAll(/@keyframes [^{]+\{(.*?)\}\./g)) {
      const pcts = [...m[1].matchAll(/([\d.]+)%\{/g)].map((x) => Number(x[1]));
      expect(pcts[0]).toBe(0);
      expect(pcts[pcts.length - 1]).toBe(100);
      for (let i = 1; i < pcts.length; i++) expect(pcts[i]).toBeGreaterThan(pcts[i - 1]);
    }
    // only what a view needs
    expect(gestureKeyframesCss(plan, ['side'])).not.toContain('-pv{');
    expect(gestureKeyframesCss(plan, ['pov'])).not.toContain('-sh{');
  });

  it("participant's choice: mirrors the figure and the hand between takes, with a fade around the cut", () => {
    const plan = gesturePlan({ hand: null, contact: 'any' });
    const css = gestureKeyframesCss(plan);
    expect(css).toContain(`.gx-k-${plan.key} .gx-p-mi{animation:gx-${plan.key}-mi 5.6s linear infinite}`);
    for (const part of ['mi', 'hm', 'fd', 'tr0', 'tr1']) expect(css).toContain(`@keyframes gx-${plan.key}-${part}{`);
    const mi = /@keyframes gx-[^{]+-mi\{(.*?)\}\./.exec(css)![1];
    expect(mi).toContain('matrix(1,0,0,1,0,0)');
    expect(mi).toContain(`matrix(-1,0,0,1,${SIDE_VIEW.w},0)`);
    // a single hand never mirrors
    expect(gestureKeyframesCss(gesturePlan({ hand: 'left', contact: 'any' }))).not.toContain('-mi{');
  });
});

// ───────────────────────────── reduced motion ─────────────────────────────

describe('reduced motion: still frames', () => {
  it('start → halfway → touching, plus the other take\'s touch', () => {
    expect(stillFrames(gesturePlan({ hand: 'right' }))).toEqual([
      { take: 0, s: 0, touch: false },
      { take: 0, s: 0.5, touch: false },
      { take: 0, s: 1, touch: true },
    ]);
    expect(stillFrames(gesturePlan({ hand: null }))[3]).toEqual({ take: 1, s: 1, touch: true });
  });

  it('step numbers sit under the frames, never on the drawing (item 12)', () => {
    const css = read('gesture.css');
    const rule = /\.gx-fnum\s*\{([^}]*)\}/.exec(css)![1];
    expect(rule).not.toMatch(/position:\s*absolute/);
    expect(css).toMatch(/\.gx-frow \.gx-frame\s*\{[^}]*flex-direction:\s*column/);
  });
});

// ───────────────────────────── CSS: layout, colours, contrast ─────────────────────────────

const UI = path.resolve(__dirname, '../src/ui');
const read = (f: string) => readFileSync(path.join(UI, f), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');

/** custom properties per theme from the top-level and the prefers-color-scheme: dark blocks */
function tokens(css: string, selector: string, theme: 'light' | 'dark'): Record<string, string> {
  const out: Record<string, string> = {};
  const grab = (text: string) => {
    for (const m of text.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      if (m[1].trim() !== selector) continue;
      for (const d of m[2].matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) out[d[1]] = d[2].trim();
    }
  };
  const dark: string[] = [];
  const top = css.replace(/@media\s*\(prefers-color-scheme:\s*dark\)\s*\{((?:[^{}]*\{[^{}]*\})*[^{}]*)\}/g, (_, body: string) => {
    dark.push(body);
    return '';
  });
  grab(top);
  if (theme === 'dark') dark.forEach(grab);
  return out;
}
function resolver(theme: 'light' | 'dark', withSession: boolean) {
  const vars: Record<string, string> = {
    ...tokens(read('styles.css'), ':root', theme),
    ...(withSession ? tokens(read('session.css'), ':root', theme) : {}),
    ...tokens(read('gesture.css'), '.gx', theme),
  };
  const resolve = (v: string): string => {
    v = v.trim();
    if (!v.startsWith('var(')) return v;
    const inner = v.slice(4, -1);
    const comma = inner.indexOf(',');
    const name = (comma < 0 ? inner : inner.slice(0, comma)).trim();
    if (vars[name] !== undefined) return resolve(vars[name]);
    if (comma < 0) throw new Error(`unresolved ${name}`);
    return resolve(inner.slice(comma + 1));
  };
  return { vars, get: (name: string, extra: Record<string, string> = {}) => resolve(extra[name] ?? vars[name]) };
}
function rgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as [number, number, number];
}
const luminance = (c: [number, number, number]) => {
  const [r, g, b] = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a: string, b: string) => {
  const la = luminance(rgb(a));
  const lb = luminance(rgb(b));
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
};
const blend = (fg: string, bg: string, alpha: number): string => {
  const f = rgb(fg);
  const g = rgb(bg);
  return `#${f.map((v, i) => Math.round(alpha * v + (1 - alpha) * g[i]).toString(16).padStart(2, '0')).join('')}`;
};

describe('styles: layout and contrast (items 8, 13, 20)', () => {
  const gesture = read('gesture.css');

  it('never stacks the two views: side by side at every width, equally tall', () => {
    expect(gesture).toMatch(/\.gx-live\s*\{[^}]*flex-wrap:\s*nowrap/);
    expect(gesture).not.toMatch(/flex-wrap:\s*wrap/);
    const grow = (cls: string) => Number(new RegExp(`\\.gx-live \\.${cls}\\s*\\{[^}]*flex:\\s*([\\d.]+)`).exec(gesture)![1]);
    // flex-grow ∝ width/height, so both panels get the same height
    expect(grow('gx-panel-side') / grow('gx-panel-pov')).toBeCloseTo(SIDE_VIEW.w / SIDE_VIEW.h, 6);
  });

  it('the "?" of the participant\'s choice is an outline, not a filled button', () => {
    expect(gesture).toMatch(/\.gx-q-badge circle\s*\{[^}]*fill:\s*none/);
  });

  for (const theme of ['light', 'dark'] as const) {
    for (const withSession of [true, false]) {
      it(`${theme} theme${withSession ? '' : ' (without session.css)'}: lines, path, touch, phone and sleeves stand out (≥ 3:1)`, () => {
        const { get } = resolver(theme, withSession);
        const stage = get('--gx-stage');
        const trailOpacity = Number(/\.gx-trail-g\s*\{[^}]*opacity:\s*([\d.]+)/.exec(gesture)![1]);
        expect(contrast(get('--gx-ink'), stage)).toBeGreaterThanOrEqual(3); // the line art on the stage
        expect(contrast(blend(get('--gx-trail'), stage, trailOpacity), stage)).toBeGreaterThanOrEqual(3);
        // the "top" arrow sits outside the phone, on the stage or over your nose
        expect(contrast(get('--gx-arrow'), stage)).toBeGreaterThanOrEqual(3);
        expect(contrast(get('--gx-arrow'), get('--gx-nose'))).toBeGreaterThanOrEqual(3);
        expect(contrast(get('--gx-muted'), stage)).toBeGreaterThanOrEqual(3);
        expect(contrast(get('--gx-phone'), stage)).toBeGreaterThanOrEqual(3);
        expect(contrast(get('--gx-back'), stage)).toBeGreaterThanOrEqual(3);
        // the burst: amber on a dark halo, so one of the two stands out on whatever is behind it
        const burst = get('--gx-burst');
        const halo = get('--gx-burst-halo');
        expect(contrast(burst, halo)).toBeGreaterThanOrEqual(3);
        for (const bg of [stage, get('--gx-screen'), get('--gx-skin'), get('--gx-nose'), get('--gx-phone')]) {
          expect(Math.max(contrast(burst, bg), contrast(halo, bg)), `burst on ${bg}`).toBeGreaterThanOrEqual(3);
        }
        // each hand's sleeve and the neutral colour of the participant's choice
        const css = tokens(gesture, '.gx.gx-hand-left', theme);
        const sl = { left: css['--gx-sleeve'], right: tokens(gesture, '.gx.gx-hand-right', theme)['--gx-sleeve'], any: tokens(gesture, '.gx.gx-hand-any', theme)['--gx-sleeve'] };
        for (const [hand, v] of Object.entries(sl)) {
          expect(v, hand).toBeDefined();
          expect(contrast(get('--gx-sleeve', { '--gx-sleeve': v }), stage), hand).toBeGreaterThanOrEqual(3);
        }
      });
    }
  }
});

// ───────────────────────────── text ─────────────────────────────

describe('gestureDescription (English fallback label)', () => {
  it('names the hand, the pace, the path and says that any part of the phone may touch', () => {
    expect(gestureDescription({ hand: 'left' })).toMatch(/^Animation\. Left hand/);
    expect(gestureDescription({ hand: null })).toContain('Either hand');
    const d = gestureDescription({ hand: 'right', speed: 'fast', contact: 'angled', variant: 'pause' });
    expect(d).toContain('quickly');
    expect(d).toContain('tilted');
    expect(d).toContain('stopping halfway');
    expect(d).toContain('any part of the phone may touch');
    expect(gestureDescription({ hand: 'right', contact: 'any' })).toContain('once with the phone upright and once tilted');
    expect(gestureDescription({ hand: 'right', orientation: 'top_left_toward' })).toContain('top to your left');
  });
});
