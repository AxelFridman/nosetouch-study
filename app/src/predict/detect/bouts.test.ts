/**
 * Hysteresis bouts (m4_detect.find_segments) on hand-made activity sequences, and the sample-by-sample detector
 * (ActivityFilter, BoutTracker, CausalDetector) against the batch functions on random streams.
 */
import { describe, expect, it } from 'vitest';
import type { Rows3, SegParams } from '../types';
import { ActivityFilter, activity } from './activity';
import { BoutTracker, CausalDetector, NOMINAL_STEP_MS, boutConstants, findBouts, isFrozen, medianStep } from './bouts';

/** lane m4's fitted SEG_GRID entry lo15_es250 (the exported bundle's segParams) */
const P: SegParams = {
  lo_dps: 15,
  hi_dps: 30,
  sustain_ms: 100,
  end_sustain_ms: 250,
  smooth_hz: 5,
  causal: true,
  min_duration_ms: 150,
  max_duration_s: 12,
  still_cap_s: 3,
  lookahead_s: 0.5,
  crop_s: 1,
};
const LAG = 44.429095814613106;
const COEFS = {
  sos: [[0.04948995626867706, 0.09897991253735412, 0.04948995626867706, 1.0, -1.2796324249978088, 0.47759225007251704]],
  zi: [[0.9505100437313232, -0.4281022938038401]],
};

const grid = (n: number): Float64Array => Float64Array.from({ length: n }, (_, i) => 0 + i * NOMINAL_STEP_MS);
/** activity made of runs: [value, count] */
const runs = (...rs: [number, number][]): number[] => rs.flatMap(([v, c]) => Array<number>(c).fill(v));

/** small seeded PRNG (mulberry32) for the property tests */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function trackAll(t: ArrayLike<number>, act: ArrayLike<number>, p: SegParams = P): ReturnType<typeof findBouts> {
  const tr = new BoutTracker(p, LAG, medianStep(t));
  const out = [];
  for (let i = 0; i < t.length; i++) out.push(...tr.push(t[i], act[i]));
  out.push(...tr.finish());
  return out;
}

describe('findBouts (m4_detect.find_segments)', () => {
  it('derives the sample counts from the grid step (15, 6, 30 at 60 Hz)', () => {
    expect(boutConstants(P, NOMINAL_STEP_MS, LAG)).toMatchObject({ nEnd: 15, nMove: 6, nLook: 30 });
  });

  it('one movement between two stillnesses: bounds, lag, stillness before / after, decision time', () => {
    const act = runs([5, 40], [50, 30], [5, 60]);
    const t = grid(act.length);
    const [b, ...rest] = findBouts(t, act, P, LAG);
    expect(rest).toEqual([]);
    expect([b.iOn, b.iEnd, b.ended, b.startedActive, b.index]).toEqual([40, 70, true, false, 0]);
    expect(b.onsetMs).toBe(t[40] - LAG);
    expect(b.endMs).toBe(t[70] - LAG);
    expect(b.decidedMs).toBe(t[70 + 30 - 1]);
    expect(b.stillBeforeS).toBe((t[39] - t[0] + medianStep(t)) / 1000);
    expect(b.stillAfterS).toBe(0.5); // capped at the look-ahead
    expect(b.peakDps).toBe(50);
  });

  it('a short stillness inside a movement does not split it; a stream that starts moving is started_active', () => {
    const act = runs([50, 20], [5, 10], [50, 20], [5, 40]);
    const [b, ...rest] = findBouts(grid(act.length), act, P, LAG);
    expect(rest).toEqual([]);
    expect([b.iOn, b.iEnd, b.startedActive, b.stillBeforeS]).toEqual([0, 50, true, 0]);
  });

  it('drops bouts without a sustained movement (> hi for 6 samples) and bouts shorter than 150 ms', () => {
    // 10 samples (167 ms) but only 5 above hi: no sustained movement
    expect(findBouts(grid(80), runs([5, 30], [50, 5], [20, 5], [5, 40]), P, LAG)).toEqual([]);
    // 10 samples (167 ms) with 6 above hi: kept
    expect(findBouts(grid(80), runs([5, 30], [20, 4], [50, 6], [5, 40]), P, LAG).length).toBe(1);
    // 8 samples above hi but the bout lasts 8 * 16.67 = 133 ms < min_duration_ms (150): dropped
    expect(findBouts(grid(80), runs([5, 30], [50, 8], [5, 42]), P, LAG)).toEqual([]);
  });

  it('the last bout of a stream can be unended (decided at the last sample, no stillness after)', () => {
    const act = runs([5, 30], [50, 40]);
    const t = grid(act.length);
    const [b] = findBouts(t, act, P, LAG);
    expect([b.ended, b.iEnd, b.stillAfterS, b.decidedMs]).toEqual([false, 69, 0, t[69]]);
  });

  it('a look-ahead cut by the end of the stream is decided at the last sample and is not frozen', () => {
    const act = runs([5, 30], [50, 30], [5, 20]);
    const t = grid(act.length);
    const [b] = findBouts(t, act, P, LAG);
    expect([b.ended, b.iEnd, b.decidedMs]).toEqual([true, 60, t[79]]);
    expect(b.stillAfterS).toBeCloseTo(20 / 60, 12);
    expect(isFrozen(b, 30, act.length, t[79] + 5000, 1000)).toBe(false); // look-ahead incomplete
  });

  it('fewer than 3 samples: no bout', () => {
    expect(findBouts(grid(2), [50, 50], P, LAG)).toEqual([]);
  });
});

describe('BoutTracker (fed sample by sample) = findBouts', () => {
  it('emits each bout when its look-ahead completes, and equals the batch on random streams', () => {
    const r = rng(20261029);
    let nBouts = 0;
    let nUnended = 0;
    let nStartedActive = 0;
    for (let trial = 0; trial < 300; trial++) {
      const n = 3 + Math.floor(r() * 600);
      const act: number[] = [];
      while (act.length < n) {
        // runs of still / moving / in-between activity of random length
        const kind = r();
        const v = kind < 0.45 ? r() * 15 : kind < 0.8 ? 30 + r() * 100 : 15 + r() * 20;
        const len = 1 + Math.floor(r() * (kind < 0.45 ? 40 : 25));
        for (let k = 0; k < len && act.length < n; k++) act.push(v + (r() - 0.5) * (kind < 0.45 ? 2 : 10));
      }
      // a slightly irregular clock (median step != 1000/60 by rounding) like a real grid far from t = 0
      const t0 = r() * 1e5;
      const t = Float64Array.from({ length: n }, (_, i) => t0 + i * NOMINAL_STEP_MS);
      const batch = findBouts(t, act, P, LAG);
      expect(trackAll(t, act), `trial ${trial}`).toEqual(batch);
      nBouts += batch.length;
      nUnended += batch.filter((b) => !b.ended).length;
      nStartedActive += batch.filter((b) => b.startedActive).length;
      // the other fitted configurations too
      const p2 = { ...P, lo_dps: 20, end_sustain_ms: 400 };
      expect(trackAll(t, act, p2), `trial ${trial} lo20_es400`).toEqual(findBouts(t, act, p2, LAG));
    }
    // the random streams exercise every branch: many bouts, unended last bouts, streams that start moving
    expect(nBouts).toBeGreaterThan(500);
    expect(nUnended).toBeGreaterThan(20);
    expect(nStartedActive).toBeGreaterThan(20);
  });

  it('exposes the stillness of the newest sample and refuses samples after finish()', () => {
    const tr = new BoutTracker(P, LAG);
    tr.push(0, 50);
    expect(tr.still).toBe(false);
    tr.push(16, 3);
    expect([tr.still, tr.lastActivity, tr.n]).toEqual([true, 3, 2]);
    tr.finish();
    expect(() => tr.push(33, 3)).toThrow();
  });
});

describe('ActivityFilter / CausalDetector (fed sample by sample) = activity() / findBouts()', () => {
  it('reproduces the batch activity bitwise, with resampling gaps (NaN rows) inside and at both ends', () => {
    const r = rng(7);
    let nBouts = 0;
    for (let trial = 0; trial < 60; trial++) {
      const n = 12 + Math.floor(r() * 500);
      const t = Float64Array.from({ length: n }, (_, i) => i * NOMINAL_STEP_MS);
      const data = new Float64Array(3 * n);
      let phase = r() * 6;
      for (let i = 0; i < n; i++) {
        phase += 0.05 + r() * 0.1;
        const amp = Math.sin(i / 37) > 0.3 ? 120 : 6;
        data[3 * i] = amp * Math.sin(phase) + (r() - 0.5) * 3;
        data[3 * i + 1] = amp * 0.5 * Math.cos(phase * 1.3) + (r() - 0.5) * 3;
        data[3 * i + 2] = (r() - 0.5) * 4;
      }
      // NaN runs: some inside, sometimes at the start or the end (all three columns, as a missing rotationRate)
      for (let g = 0; g < 3; g++) {
        const at = r() < 0.2 ? 0 : r() < 0.2 ? n - 5 : Math.floor(r() * n);
        const len = 1 + Math.floor(r() * 25);
        for (let i = at; i < Math.min(n, at + len); i++) data.fill(NaN, 3 * i, 3 * i + 3);
      }
      const gyro: Rows3 = { n, data };
      const batch = activity(t, gyro, P, COEFS);
      const f = new ActivityFilter(P, COEFS);
      const got: number[] = [];
      for (let i = 0; i < n; i++) for (const s of f.push(t[i], data[3 * i], data[3 * i + 1], data[3 * i + 2])) got.push(s.act);
      for (const s of f.finish()) got.push(s.act);
      expect(got, `trial ${trial}`).toEqual(Array.from(batch));
      expect(f.count).toBe(n);
      // and the bouts
      const cd = new CausalDetector(P, COEFS, LAG, medianStep(t));
      const bouts = [];
      for (let i = 0; i < n; i++) bouts.push(...cd.push(t[i], data[3 * i], data[3 * i + 1], data[3 * i + 2]));
      bouts.push(...cd.finish());
      expect(bouts, `trial ${trial} bouts`).toEqual(findBouts(t, batch, P, LAG));
      nBouts += bouts.length;
    }
    expect(nBouts).toBeGreaterThan(30);
  });

  it('a column that is never finite gives activity 0 (fill_gaps_linear leaves it NaN)', () => {
    const n = 20;
    const t = grid(n);
    const gyro: Rows3 = { n, data: new Float64Array(3 * n).fill(NaN) };
    expect(Array.from(activity(t, gyro, P, COEFS)).every((v) => v === 0)).toBe(true);
    const f = new ActivityFilter(P, COEFS);
    for (let i = 0; i < n; i++) expect(f.push(t[i], NaN, NaN, NaN)).toEqual([]);
    expect(f.finish().map((s) => s.act)).toEqual(Array<number>(n).fill(0));
  });

  it('streams shorter than 12 samples are left unfiltered by the batch function (m4_detect.py:110)', () => {
    const n = 11;
    const data = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) data[3 * i] = i * 10;
    expect(Array.from(activity(grid(n), { n, data }, P, COEFS))).toEqual(Array.from({ length: n }, (_, i) => i * 10));
  });
});
