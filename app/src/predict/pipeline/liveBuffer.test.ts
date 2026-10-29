/**
 * LiveBuffer = the collector's recorder: the same samples pushed into a LiveBuffer and into
 * core/recorder.ts StreamRecorder (t0 = the first sample) give the same RawMotion, bit for bit
 * (APP_SPEC 1.3), including missing vectors, missing components, -0 and values beyond the rounding range.
 */
import { describe, expect, it } from 'vitest';
import { StreamRecorder } from '../../core/recorder';
import type { MotionSample, Vec3 } from '../../core/types';
import type { RawMotion } from '../types';
import { createLiveBuffer } from './liveBuffer';
import { rawMotionFromRecord } from './record';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function samples(count: number, seed: number): MotionSample[] {
  const rnd = mulberry32(seed);
  const out: MotionSample[] = [];
  let t = 81234.56789 + rnd() * 1000;
  const vec = (scale: number): Vec3 | null => {
    const u = rnd();
    if (u < 0.03) return null;
    const v: Vec3 = [(rnd() - 0.5) * scale, (rnd() - 0.5) * scale, (rnd() - 0.5) * scale];
    if (u < 0.05) v[Math.floor(rnd() * 3)] = NaN; // one missing component
    if (u > 0.98) v[0] = -0;
    if (u > 0.995) v[1] = 1e17; // beyond 2^52 at this scale: stored as is
    return v;
  };
  for (let i = 0; i < count; i++) {
    t += 8 + rnd() * 12 + rnd() * 1e-3;
    const rr = vec(400);
    out.push({
      t,
      ts: t - 5,
      acc: vec(4),
      accg: vec(25),
      rr: rr ? { alpha: rr[0], beta: rr[1], gamma: rr[2] } : null,
      interval: 16,
    });
  }
  return out;
}

function expectSameRaw(a: RawMotion, b: RawMotion): void {
  expect(a.t.length).toBe(b.t.length);
  const eq = (x: ArrayLike<number>, y: ArrayLike<number>, what: string): void => {
    expect(x.length, what).toBe(y.length);
    for (let i = 0; i < x.length; i++) {
      const ok = Object.is(x[i], y[i]) || (Number.isNaN(x[i]) && Number.isNaN(y[i]));
      if (!ok) expect(`${what}[${i}] = ${x[i]}`).toBe(`${what}[${i}] = ${y[i]}`);
    }
  };
  eq(a.t, b.t, 't');
  eq(a.acc.data, b.acc.data, 'acc');
  eq(a.accg.data, b.accg.data, 'accg');
  eq(a.rr.data, b.rr.data, 'rr');
}

describe('LiveBuffer', () => {
  it('stores samples exactly as StreamRecorder.addMotionRel (t relative to the first sample)', () => {
    for (const seed of [1, 2, 3]) {
      const ss = samples(1500, seed);
      const rec = new StreamRecorder(ss[0].t);
      const buf = createLiveBuffer();
      for (const s of ss) {
        rec.addMotionRel(s);
        expect(buf.push(s)).toBe(buf.n - 1);
      }
      expect(buf.t0).toBe(ss[0].t);
      const fromRecord = rawMotionFromRecord(rec.build().motion);
      expectSameRaw(buf.raw(), fromRecord);
      expect(buf.raw().t[0]).toBe(0);
    }
  });

  it('rejects a sample whose time is not finite or not after the previous stored time', () => {
    const buf = createLiveBuffer();
    const base: MotionSample = { t: 1000, ts: null, acc: [0, 0, 0], accg: [0, 9.8, 0], rr: { alpha: 1, beta: 2, gamma: 3 }, interval: null };
    expect(buf.push({ ...base, t: NaN })).toBe(-1);
    expect(buf.t0).toBe(null);
    expect(buf.push(base)).toBe(0);
    expect(buf.push({ ...base, t: 1000.004 })).toBe(-1); // rounds to the same 0.00 ms
    expect(buf.push({ ...base, t: 999 })).toBe(-1);
    expect(buf.push({ ...base, t: Infinity })).toBe(-1);
    expect(buf.push({ ...base, t: 1016.6666 })).toBe(1);
    expect(buf.n).toBe(2);
    expect(Array.from(buf.raw().t)).toEqual([0, 16.67]);
  });

  it('raw(n) is a prefix view that stays valid while the buffer grows and after clear()', () => {
    const ss = samples(3000, 9);
    const buf = createLiveBuffer();
    ss.slice(0, 100).forEach((s) => buf.push(s));
    const early = buf.raw(50);
    const copy = { t: early.t.slice(), acc: early.acc.data.slice() };
    ss.slice(100).forEach((s) => buf.push(s)); // grows past the initial capacity several times
    expect(buf.raw(50).t.length).toBe(50);
    expect(Array.from(early.t)).toEqual(Array.from(copy.t));
    expect(Object.is(early.acc.data[0], copy.acc[0]) || Number.isNaN(copy.acc[0])).toBe(true);
    buf.clear();
    expect(buf.n).toBe(0);
    expect(buf.t0).toBe(null);
    buf.push(ss[0]);
    expect(Array.from(early.t)).toEqual(Array.from(copy.t));
    expect(buf.raw(10).t.length).toBe(1);
  });
});
