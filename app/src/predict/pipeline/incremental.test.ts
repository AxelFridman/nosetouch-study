/**
 * Incremental use on a growing buffer gives the same numbers as one batch call:
 *  - createStreamBuilder(norm).update(prefix) equals streamFrom(prefix, norm) bit for bit at every prefix (one row
 *    at a time and in random chunks), on the fixtures and on generated edge cases (gaps longer than 100 ms, missing
 *    columns, isolated and repeated snapshots, runs of 1, 2, 5, 15, 16 and 17 values, a missing accelerometer);
 *  - a live replay (LiveSample -> LiveBuffer -> builder, updated every few samples) ends with exactly the stream of
 *    the stored record, i.e. the Python reference's stream;
 *  - the work per update stays bounded (the backward pass converges instead of re-running the whole run).
 */
import { describe, expect, it } from 'vitest';
import type { MotionSample } from '../../core/types';
import type { RawMotion, Stream, StreamNormalization } from '../types';
import { REAL_DIR, SYNTH_DIR, fixturesPresent, loadFixtures, rowsOf, type Fixture } from './fixtures.testutil';
import { createIncrementalStreamFrom, createStreamBuilder } from './incremental';
import { createLiveBuffer } from './liveBuffer';
import { productionNormalization } from './normalize';
import { rawMotionFromRecord, rawPrefix } from './record';
import { PipelineError, streamFrom } from './standardize';

function batch(raw: RawMotion, norm: StreamNormalization): Stream | null {
  try {
    return streamFrom(raw, norm);
  } catch (e) {
    if (e instanceof PipelineError && e.code === 'not_enough_data') return null;
    throw e;
  }
}

function sameArray(a: ArrayLike<number>, b: ArrayLike<number>): string | null {
  if (a.length !== b.length) return `length ${a.length} != ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    if (!(Object.is(a[i], b[i]) || (Number.isNaN(a[i]) && Number.isNaN(b[i])))) return `[${i}] ${a[i]} != ${b[i]}`;
  }
  return null;
}

/** null when equal, else a description of the first difference */
function streamDiff(a: Stream | null, b: Stream | null): string | null {
  if (a === null || b === null) return a === b ? null : `null vs stream`;
  const checks: [string, ArrayLike<number>, ArrayLike<number>][] = [
    ['t', a.t, b.t],
    ['f', a.f.data, b.f.data],
    ['acc', a.acc.data, b.acc.data],
    ['gyro', a.gyro.data, b.gyro.data],
    ['valid', a.valid, b.valid],
    ['tRaw', a.tRaw, b.tRaw],
  ];
  for (const [name, x, y] of checks) {
    const d = sameArray(x, y);
    if (d) return `${name} ${d}`;
  }
  if (a.repeatsDropped !== b.repeatsDropped) return `repeatsDropped ${a.repeatsDropped} != ${b.repeatsDropped}`;
  if (a.f.n !== b.f.n || a.engine !== b.engine) return 'shape or engine';
  return null;
}

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

/** Feeds prefixes ending at `cuts` and compares every update with the batch result; returns the builder stats. */
function replay(raw: RawMotion, norm: StreamNormalization, cuts: number[]): ReturnType<typeof createStreamBuilder>['stats'] {
  const b = createStreamBuilder(norm);
  for (const k of cuts) {
    const p = rawPrefix(raw, k);
    const d = streamDiff(b.update(p), batch(p, norm));
    if (d) throw new Error(`prefix ${k}: ${d}`);
  }
  return b.stats;
}

function everyPrefix(n: number, step = 1): number[] {
  const out: number[] = [];
  for (let k = 1; k <= n; k += step) out.push(k);
  if (out[out.length - 1] !== n) out.push(n);
  return out;
}

function randomCuts(n: number, seed: number): number[] {
  const rnd = mulberry32(seed);
  const out: number[] = [];
  let k = 0;
  while (k < n) {
    k = Math.min(n, k + 1 + Math.floor(rnd() * 12));
    out.push(k);
  }
  return out;
}

/** a generated raw buffer with every situation the incremental logic must get right */
function edgeCaseRaw(seed: number): RawMotion {
  const rnd = mulberry32(seed);
  const t: number[] = [];
  const acc: number[] = [];
  const accg: number[] = [];
  const rr: number[] = [];
  let tt = 0;
  let phase = 0;
  const push = (gap: number, opts: { repeat?: boolean; nanCol?: number; noAcc?: boolean } = {}): void => {
    tt = Math.round((tt + gap) * 100) / 100;
    t.push(tt);
    const k = t.length - 1;
    phase += 0.07;
    let a = [Math.sin(phase) * 0.8, Math.cos(phase * 0.7) * 0.5, rnd() * 0.2];
    let g = [Math.sin(phase * 0.3) * 3 + rnd() * 0.05, 9.4 + rnd() * 0.1, Math.cos(phase * 0.2) * 2];
    let w = [Math.sin(phase) * 60 + rnd(), Math.cos(phase * 1.3) * 45, (rnd() - 0.5) * 30];
    if (opts.repeat && k > 0) {
      a = acc.slice(3 * k - 3, 3 * k);
      g = accg.slice(3 * k - 3, 3 * k);
      w = rr.slice(3 * k - 3, 3 * k);
    }
    const r4 = (v: number): number => Math.round(v * 1e4) / 1e4;
    const r3 = (v: number): number => Math.round(v * 1e3) / 1e3;
    acc.push(...a.map((v) => (opts.noAcc ? NaN : opts.repeat ? v : r4(v))));
    accg.push(...g.map((v) => (opts.repeat ? v : r4(v))));
    rr.push(...w.map((v) => (opts.repeat ? v : r3(v))));
    if (opts.nanCol !== undefined) {
      const c = opts.nanCol;
      if (c < 3) accg[3 * k + c] = NaN;
      else if (c < 6) acc[3 * k + c - 3] = NaN;
      else rr[3 * k + c - 6] = NaN;
    }
  };
  const step = (): number => 15 + rnd() * 3;
  for (let i = 0; i < 40; i++) push(step());
  push(16, { repeat: true }); // an isolated repeat
  for (let i = 0; i < 5; i++) push(step());
  push(16, { repeat: true });
  push(16, { repeat: true }); // a run of three equal rows (kept)
  for (let i = 0; i < 30; i++) push(step());
  push(140); // a gap > 100 ms: a new run starts
  for (let i = 0; i < 1; i++) push(step()); // a run of 1-2 values between gaps
  push(130);
  for (let i = 0; i < 4; i++) push(step()); // a run of about 5 values
  push(160);
  for (let i = 0; i < 14; i++) push(step()); // about 15 values
  push(120);
  for (let i = 0; i < 16; i++) push(step()); // about 17 values
  push(150);
  for (let i = 0; i < 50; i++) push(step(), { nanCol: i % 17 === 3 ? (i % 9) : undefined }); // single missing cells
  for (let i = 0; i < 12; i++) push(step(), { nanCol: 7 }); // one gyro column missing for 12 rows (> 100 ms)
  for (let i = 0; i < 30; i++) push(step(), { noAcc: true }); // no linear acceleration
  for (let i = 0; i < 25; i++) push(step());
  push(16, { repeat: true }); // the newest row an isolated repeat ...
  push(16, { repeat: true }); // ... that the next row turns into a kept run
  for (let i = 0; i < 20; i++) push(step());
  push(16, { repeat: true }); // ends on a dropped repeat (WebKit)
  const n = t.length;
  return {
    t: Float64Array.from(t),
    acc: { n, data: Float64Array.from(acc) },
    accg: { n, data: Float64Array.from(accg) },
    rr: { n, data: Float64Array.from(rr) },
  };
}

function liveReplay(fx: Fixture, every: number): Stream | null {
  const norm = productionNormalization(fx.engine);
  const buf = createLiveBuffer();
  const b = createStreamBuilder(norm);
  const m = fx.motion;
  const origin = 52341.123; // a source clock that is not the record's
  let last: Stream | null = null;
  const v = (x: number | null): number => (x === null ? NaN : x);
  for (let i = 0; i < m.t.length; i++) {
    const s: MotionSample = {
      t: origin + m.t[i],
      ts: null,
      acc: [v(m.acc_x[i]), v(m.acc_y[i]), v(m.acc_z[i])],
      accg: [v(m.accg_x[i]), v(m.accg_y[i]), v(m.accg_z[i])],
      rr: { alpha: v(m.rr_alpha[i]), beta: v(m.rr_beta[i]), gamma: v(m.rr_gamma[i]) },
      interval: 16,
    };
    expect(buf.push(s)).toBe(i);
    if (i % every === every - 1) last = b.update(buf.raw());
  }
  return b.update(buf.raw()) ?? last;
}

describe('incremental stream = batch stream', () => {
  it('generated edge cases, every prefix, every engine rule', () => {
    for (const [seed, engine] of [
      [11, 'webkit'],
      [12, 'blink'],
      [13, 'gecko'],
    ] as const) {
      const raw = edgeCaseRaw(seed);
      const norm = productionNormalization(engine);
      const st = replay(raw, norm, everyPrefix(raw.t.length));
      expect(st.batchOnly).toBe(false); // the incremental path itself, not the batch fallback
      expect(st.converged).toBeGreaterThan(0);
      replay(raw, norm, randomCuts(raw.t.length, seed));
      const full = batch(raw, norm)!;
      expect(full.repeatsDropped).toBeGreaterThan(0);
      expect(Array.from(full.valid).some((x) => x === 0)).toBe(true);
    }
  }, 300_000);

  it.skipIf(!fixturesPresent(SYNTH_DIR))('synthetic fixtures: every prefix and random chunks; the live replay ends on the reference stream', () => {
    for (const fx of loadFixtures(SYNTH_DIR)) {
      const raw = rawMotionFromRecord(fx.motion);
      const norm = productionNormalization(fx.engine);
      const st = replay(raw, norm, everyPrefix(raw.t.length));
      replay(raw, norm, randomCuts(raw.t.length, fx.id.length));
      // bounded work: one row per update, the backward pass stops after about 100 samples instead of the run
      const perUpdatePerColumn = st.backwardSteps / Math.max(1, raw.t.length) / 9;
      expect(perUpdatePerColumn).toBeLessThan(200);
      expect(st.converged).toBeGreaterThan(0);
      const live = liveReplay(fx, 6)!;
      const ref = fx.reference.stream;
      expect(sameArray(live.t, ref.t)).toBeNull();
      expect(sameArray(live.f.data, rowsOf(ref.f))).toBeNull();
      expect(sameArray(live.gyro.data, rowsOf(ref.gyro))).toBeNull();
      expect(sameArray(live.acc.data, rowsOf(ref.acc))).toBeNull();
      expect(live.repeatsDropped).toBe(fx.reference.repeats_dropped);
    }
  }, 300_000);

  it.skipIf(!fixturesPresent(REAL_DIR))('local real fixtures: strided prefixes and random chunks; the live replay ends on the reference stream', () => {
    let fixtures = 0;
    for (const fx of loadFixtures(REAL_DIR)) {
      fixtures++;
      const raw = rawMotionFromRecord(fx.motion);
      const norm = productionNormalization(fx.engine);
      replay(raw, norm, everyPrefix(raw.t.length, 7));
      replay(raw, norm, randomCuts(raw.t.length, fixtures));
      const live = liveReplay(fx, 6)!;
      const ref = fx.reference.stream;
      expect(sameArray(live.t, ref.t)).toBeNull();
      expect(sameArray(live.f.data, rowsOf(ref.f))).toBeNull();
      expect(sameArray(live.gyro.data, rowsOf(ref.gyro))).toBeNull();
    }
    expect(fixtures).toBeGreaterThan(0);
  }, 600_000);

  it('createIncrementalStreamFrom is a drop-in streamFrom for a growing buffer (same numbers, same error)', () => {
    const raw = edgeCaseRaw(31);
    const sf = createIncrementalStreamFrom();
    const webkit = productionNormalization('webkit');
    expect(() => sf(rawPrefix(raw, 4), webkit)).toThrow(PipelineError);
    for (const k of randomCuts(raw.t.length, 5).filter((x) => x >= 5)) {
      expect(streamDiff(sf(rawPrefix(raw, k), webkit), batch(rawPrefix(raw, k), webkit))).toBeNull();
    }
    const blink = productionNormalization('blink'); // another normalisation restarts it
    expect(streamDiff(sf(raw, blink), batch(raw, blink))).toBeNull();
  });

  it('restarts when the buffer does not extend the previous one, and falls back to batch on unordered times', () => {
    const raw = edgeCaseRaw(21);
    const norm = productionNormalization('blink');
    const b = createStreamBuilder(norm);
    expect(streamDiff(b.update(rawPrefix(raw, 200)), batch(rawPrefix(raw, 200), norm))).toBeNull();
    expect(streamDiff(b.update(rawPrefix(raw, 120)), batch(rawPrefix(raw, 120), norm))).toBeNull(); // shorter: restart
    const other = edgeCaseRaw(22);
    expect(streamDiff(b.update(rawPrefix(other, 150)), batch(rawPrefix(other, 150), norm))).toBeNull(); // different rows
    expect(b.update(rawPrefix(other, 3))).toBeNull(); // fewer than 5 rows: not enough data
    // a cleared and refilled buffer whose times happen to equal the old ones (every stream starts at t = 0)
    const sameTimes: RawMotion = { ...other, t: raw.t.slice(0, other.t.length) };
    expect(streamDiff(b.update(rawPrefix(raw, 200)), batch(rawPrefix(raw, 200), norm))).toBeNull();
    expect(streamDiff(b.update(rawPrefix(sameTimes, 260)), batch(rawPrefix(sameTimes, 260), norm))).toBeNull();
    const t = raw.t.slice(0, 60);
    t[30] = t[29]; // a repeated time (never from a LiveBuffer): batch fallback, still identical
    const odd: RawMotion = { ...rawPrefix(raw, 60), t };
    const c = createStreamBuilder(norm);
    expect(streamDiff(c.update(odd), batch(odd, norm))).toBeNull();
    expect(c.stats.batchOnly).toBe(true);
  });
});
