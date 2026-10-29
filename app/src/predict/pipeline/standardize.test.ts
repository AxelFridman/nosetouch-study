/**
 * Parity of the pipeline with the Python reference (analysis/stage3 reference_predict, which runs
 * standardize.stream_from_record and m4_detect.window_for themselves) on the golden fixtures:
 *   raw motion -> streamFrom: grid t, valid, repeats dropped, f / acc / gyro;
 *   every segment's bounds -> windowFor (crop + canonical + time_normalised): crop indices, gravity_v1
 *   metadata, the (17, 64) float32 window; once from our own stream (end to end) and once from the Python
 *   stream (canonical alone).
 * Committed synthetic fixtures always; the local real-recording fixtures only where the git-ignored folder exists.
 * Tolerances (PORT_SPEC 5.6): stream 1e-9 (expected about 1e-13), windows 1 float32 ulp, gravity 1e-9.
 */
import { describe, expect, it } from 'vitest';
import type { Stream } from '../types';
import { Diff, REAL_DIR, SYNTH_DIR, fixturesPresent, loadFixtures, num, rowsOf, ulp32, type Fixture } from './fixtures.testutil';
import { productionNormalization } from './normalize';
import { norm3Dot } from './numeric';
import { rawMotionFromRecord } from './record';
import { PipelineError, canonical, cropStream, streamFrom, timeNormalised, windowFor } from './standardize';

interface Report {
  fixtures: number;
  segments: number;
  windows: number;
  stream: { t: Diff; f: Diff; acc: Diff; gyro: Diff };
  validMismatch: number;
  gridMismatch: number;
  repeatsMismatch: number;
  cropMismatch: number;
  gravityCountMismatch: number;
  closure: Diff;
  magnitude: Diff;
  window: Diff;
  windowFromRef: Diff;
  windowUlpMax: number;
  windowUlpMaxFromRef: number;
  windowValuesOff: number;
  canonicalFailMismatch: number;
}

function referenceStream(fx: Fixture): Stream {
  const s = fx.reference.stream;
  const n = s.t.length;
  return {
    t: Float64Array.from(s.t),
    f: { n, data: rowsOf(s.f) },
    acc: { n, data: rowsOf(s.acc) },
    gyro: { n, data: rowsOf(s.gyro) },
    valid: Uint8Array.from(s.valid, (v) => (v ? 1 : 0)),
    fsHz: 60,
    engine: null,
    repeatsDropped: fx.reference.repeats_dropped,
    tRaw: new Float64Array(0),
    gravityMagnitude: null,
    baselineGravity: null,
  };
}

function check(fixtures: Fixture[]): Report {
  const rep: Report = {
    fixtures: fixtures.length,
    segments: 0,
    windows: 0,
    stream: { t: new Diff(), f: new Diff(), acc: new Diff(), gyro: new Diff() },
    validMismatch: 0,
    gridMismatch: 0,
    repeatsMismatch: 0,
    cropMismatch: 0,
    gravityCountMismatch: 0,
    closure: new Diff(),
    magnitude: new Diff(),
    window: new Diff(),
    windowFromRef: new Diff(),
    windowUlpMax: 0,
    windowUlpMaxFromRef: 0,
    windowValuesOff: 0,
    canonicalFailMismatch: 0,
  };
  for (const fx of fixtures) {
    const norm = productionNormalization(fx.engine);
    expect(norm).toEqual(fx.reference.normalization);
    const raw = rawMotionFromRecord(fx.motion);
    const st = streamFrom(raw, norm);
    const ref = fx.reference.stream;
    if (st.t.length !== fx.reference.n_grid) rep.gridMismatch++;
    if (st.repeatsDropped !== fx.reference.repeats_dropped) rep.repeatsMismatch++;
    rep.stream.t.addAll(st.t, ref.t);
    rep.stream.f.addAll(st.f.data, rowsOf(ref.f));
    rep.stream.acc.addAll(st.acc.data, rowsOf(ref.acc));
    rep.stream.gyro.addAll(st.gyro.data, rowsOf(ref.gyro));
    ref.valid.forEach((v, i) => {
      if ((st.valid[i] === 1) !== v) rep.validMismatch++;
    });
    const refStream = referenceStream(fx);
    for (const sg of fx.reference.segments) {
      rep.segments++;
      if (!sg.canonical_ok) {
        // Python's canonical raised: so must ours
        let threw = false;
        try {
          windowFor(st, sg.onset_ms, sg.end_ms, 1.0, 0.5);
        } catch (e) {
          threw = e instanceof PipelineError;
        }
        if (!threw) rep.canonicalFailMismatch++;
        continue;
      }
      rep.windows++;
      const { window, can } = windowFor(st, sg.onset_ms, sg.end_ms, 1.0, 0.5);
      const c = sg.crop;
      if (can.n !== c.n || can.iOn !== c.i_on || can.iEnd !== c.i_end || !Object.is(can.t[0], c.t0_ms)) rep.cropMismatch++;
      const gr = sg.gravity!;
      if (can.restPreN !== gr.pre_n || can.restPostN !== gr.post_n || can.gravityMethod !== gr.method) rep.gravityCountMismatch++;
      rep.closure.add(can.closureDeg, num(gr.closure_deg));
      // the reference reports np.linalg.norm(can.g[can.i_on]) (one vector: BLAS ddot), not the scale factor
      const g = can.g.data;
      rep.magnitude.add(norm3Dot(g[3 * can.iOn], g[3 * can.iOn + 1], g[3 * can.iOn + 2]), num(gr.magnitude));
      const want = sg.window!;
      // the same window from the Python stream: canonical + time_normalised alone
      const crop2 = cropStream(refStream, sg.onset_ms - 1000.0, sg.end_ms + 500.0);
      const w2 = timeNormalised(canonical(crop2, sg.onset_ms, sg.end_ms), sg.onset_ms, sg.end_ms);
      for (let ch = 0; ch < 17; ch++) {
        for (let k = 0; k < 64; k++) {
          const exp = num(want[ch][k]);
          const got = window.data[ch * 64 + k];
          const got2 = w2.data[ch * 64 + k];
          rep.window.add(got, exp);
          rep.windowFromRef.add(got2, exp);
          const u = ulp32(got, exp);
          if (u > 0) rep.windowValuesOff++;
          rep.windowUlpMax = Math.max(rep.windowUlpMax, u);
          rep.windowUlpMaxFromRef = Math.max(rep.windowUlpMaxFromRef, ulp32(got2, exp));
        }
      }
    }
  }
  return rep;
}

function summary(name: string, r: Report): string {
  return [
    `${name}: ${r.fixtures} fixtures, ${r.segments} segments, ${r.windows} windows`,
    `  stream t ${r.stream.t}`,
    `  stream f ${r.stream.f}`,
    `  stream acc ${r.stream.acc}`,
    `  stream gyro ${r.stream.gyro}`,
    `  valid mismatches ${r.validMismatch}, grid-length mismatches ${r.gridMismatch}, repeats mismatches ${r.repeatsMismatch}`,
    `  crop (n, t0, i_on, i_end) mismatches ${r.cropMismatch}; gravity (pre_n, post_n, method) mismatches ${r.gravityCountMismatch}`,
    `  closure_deg ${r.closure}`,
    `  gravity magnitude ${r.magnitude}`,
    `  window (end to end) ${r.window}; max ${r.windowUlpMax} float32 ulp; values off ${r.windowValuesOff}`,
    `  window (from the Python stream) ${r.windowFromRef}; max ${r.windowUlpMaxFromRef} float32 ulp`,
    `  canonical failures not reproduced ${r.canonicalFailMismatch}`,
  ].join('\n');
}

function assertReport(r: Report): void {
  expect(r.gridMismatch).toBe(0);
  expect(r.repeatsMismatch).toBe(0);
  expect(r.validMismatch).toBe(0);
  expect(r.stream.t.notBitwise).toBe(0);
  for (const d of [r.stream.f, r.stream.acc, r.stream.gyro]) {
    expect(d.nanMismatch).toBe(0);
    expect(d.max).toBeLessThanOrEqual(1e-9);
  }
  expect(r.cropMismatch).toBe(0);
  expect(r.gravityCountMismatch).toBe(0);
  expect(r.closure.nanMismatch).toBe(0);
  expect(r.closure.max).toBeLessThanOrEqual(1e-9);
  expect(r.magnitude.max).toBeLessThanOrEqual(1e-9);
  expect(r.window.nanMismatch).toBe(0);
  expect(r.windowUlpMax).toBeLessThanOrEqual(1);
  expect(r.windowUlpMaxFromRef).toBeLessThanOrEqual(1);
  expect(r.canonicalFailMismatch).toBe(0);
}

describe.skipIf(!fixturesPresent(SYNTH_DIR))('pipeline parity: committed synthetic fixtures', () => {
  it('stream, crops, gravity and windows equal the Python reference', () => {
    const r = check(loadFixtures(SYNTH_DIR));
    console.log(summary('synthetic', r));
    expect(r.fixtures).toBeGreaterThanOrEqual(5);
    expect(r.windows).toBeGreaterThan(0);
    assertReport(r);
  });
});

describe.skipIf(!fixturesPresent(REAL_DIR))('pipeline parity: local real-recording fixtures (git-ignored)', () => {
  it('stream, crops, gravity and windows equal the Python reference', () => {
    const r = check(loadFixtures(REAL_DIR));
    console.log(summary('real (local)', r));
    expect(r.fixtures).toBeGreaterThan(0);
    assertReport(r);
  });
});

describe('streamFrom and canonical edge cases', () => {
  it('fewer than 5 kept rows is not enough data, as in Python', () => {
    const motion = {
      t: [0, 16, 33, 50],
      acc_x: [0, 0, 0, 0],
      acc_y: [0, 0, 0, 0],
      acc_z: [0, 0, 0, 0],
      accg_x: [0, 0.1, 0.2, 0.3],
      accg_y: [9.8, 9.8, 9.8, 9.8],
      accg_z: [0, 0, 0, 0],
      rr_alpha: [1, 2, 3, 4],
      rr_beta: [0, 0, 0, 0],
      rr_gamma: [0, 0, 0, 0],
    };
    expect(() => streamFrom(rawMotionFromRecord(motion), productionNormalization('blink'))).toThrow(PipelineError);
  });

  it('canonical rejects reversed bounds and bounds holding fewer than 3 grid samples', () => {
    const n = 40;
    const st: Stream = {
      t: Float64Array.from({ length: n }, (_, i) => i * (1000 / 60)),
      f: { n, data: Float64Array.from({ length: 3 * n }, (_, k) => (k % 3 === 1 ? 9.8 : 0.1)) },
      acc: { n, data: new Float64Array(3 * n) },
      gyro: { n, data: Float64Array.from({ length: 3 * n }, (_, k) => (k % 3 === 2 ? 5 : 0)) },
      valid: new Uint8Array(n).fill(1),
      fsHz: 60,
      engine: null,
      repeatsDropped: 0,
      tRaw: new Float64Array(0),
      gravityMagnitude: null,
      baselineGravity: null,
    };
    expect(() => canonical(st, 300, 200)).toThrow(PipelineError);
    expect(() => canonical(st, 300, NaN)).toThrow(PipelineError);
    expect(() => canonical(st, 300, 330)).toThrow(PipelineError);
    const can = canonical(st, 200, 400);
    expect(can.iOn).toBe(12);
    expect(can.iEnd).toBe(24);
    const w = timeNormalised(can, 200, 400);
    expect(w.data.length).toBe(17 * 64);
    expect(w.data[15 * 64]).toBe(0); // yaw_cum re-zeroed at the onset
  });
});
