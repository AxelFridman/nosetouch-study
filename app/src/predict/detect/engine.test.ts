/**
 * The live session engine on the committed synthetic fixtures, replayed sample by sample through a stand-in
 * pipeline: the LiveBuffer keeps the sample times and streamFrom serves the fixture's own standardised stream up to
 * the newest sample (exact for every frozen bout: after the 1 s settle the zero-phase filter no longer changes what a
 * bout reads). Checked against analysis/stage3/reference.py's live rule evaluated at the engine's last tick.
 */
import { describe, expect, it } from 'vitest';
import type { DetectorBundle, EngineSnapshot, HandModel, LiveBuffer, LiveSample, ModelAttemptOutput, RawMotion, Stream } from '../types';
import { cropStream, detectWindowPipeline } from './canonical';
import { makeCreateSessionEngine, type EngineDeps } from './engine';
import { stopOnCurve } from './stopping';
import { REAL_DIR, SYNTH_DIR, loadDetector, loadFixtures, m4TestModel, realFixturesPresent, streamOf, type Fixture } from './testSupport';

const det: DetectorBundle = loadDetector();
const fixtures = loadFixtures(SYNTH_DIR);

function fakeBuffer(): LiveBuffer {
  let t0: number | null = null;
  const ts: number[] = [];
  const rows = (n: number) => ({ n, data: new Float64Array(3 * n) });
  return {
    push(s: LiveSample): number {
      if (!Number.isFinite(s.t)) return -1;
      if (t0 === null) t0 = s.t;
      const rel = s.t - t0;
      if (ts.length && !(rel > ts[ts.length - 1])) return -1;
      ts.push(rel);
      return ts.length - 1;
    },
    get n() {
      return ts.length;
    },
    get t0() {
      return t0;
    },
    raw(n = ts.length): RawMotion {
      return { t: Float64Array.from(ts.slice(0, n)), acc: rows(n), accg: rows(n), rr: rows(n) };
    },
    clear() {
      t0 = null;
      ts.length = 0;
    },
  };
}

/** a pipeline stand-in serving the fixture's stream; records the last grid time it served */
function fakePipeline(stream: Stream): { deps: EngineDeps; lastT: () => number } {
  let last = NaN;
  const deps: EngineDeps = {
    detector: det,
    pipeline: {
      ...detectWindowPipeline,
      createLiveBuffer: fakeBuffer,
      streamFrom: (raw: RawMotion) => {
        if (raw.t.length < 5) throw new RangeError('fewer than 5 rows');
        const s = cropStream(stream, -Infinity, raw.t[raw.t.length - 1]);
        last = s.t[s.t.length - 1];
        return s;
      },
    },
  };
  return { deps, lastT: () => last };
}

function samplesOf(fx: Fixture, offset = 0): LiveSample[] {
  return fx.reference.stream.t.map((t) => ({ t: t + offset, ts: null, acc: null, accg: [0, 0, 9.8], rr: { alpha: 0, beta: 0, gamma: 0 }, interval: 16 }));
}

/** the yaw-reference-like aggregation (mean P) over m4's outputs: a second model to switch to */
function meanPModel(base: HandModel): HandModel {
  return {
    spec: { id: 'yaw' } as HandModel['spec'],
    predict: (input) => ({ ...base.predict(input), modelId: 'yaw' }),
    aggregate: (o: readonly ModelAttemptOutput[]) => (o.length ? o.reduce((s, x) => s + x.pRight, 0) / o.length : 0.5),
  };
}

function replay(fx: Fixture, chunks: (i: number) => number = () => 1, maxStreamMs?: number) {
  const { deps, lastT } = fakePipeline(streamOf(fx));
  const engine = makeCreateSessionEngine(deps)({
    normalization: { gravitySign: 1, rrVerdict: 'w3c', rrUnits: 'deg_per_s', engine: 'blink', quantizeIos: false, source: 'engine_prior' },
    models: [m4TestModel()],
    driver: 'm4',
    maxStreamMs,
  });
  const snaps: EngineSnapshot[] = [];
  engine.subscribe((s) => snaps.push(s));
  const xs = samplesOf(fx, 12345.678);
  for (let i = 0; i < xs.length; ) {
    const k = Math.max(1, chunks(i));
    engine.push(xs.slice(i, i + k));
    i += k;
  }
  return { engine, snaps, lastT: lastT() };
}

/** what the reference's live rule gives at the engine's last tick time tl (m4 driver) */
function expected(fx: Fixture, tl: number) {
  const ref = fx.reference;
  const frozenAt = (k: number) => ref.segments[k].ended && ref.segments[k].freeze_after_ms <= tl;
  const grip = ref.grip.grip_warning && ref.segments.some((_, k) => frozenAt(k));
  const att = ref.attempts.filter(frozenAt);
  const kc = ref.decisions.m4.k_curve.slice(0, att.length);
  const dec = stopOnCurve('m4', kc.map((e) => e.p), kc.map((e) => e.decided_ms), det.stopping, false);
  return { grip, att, dec };
}

/** replays one fixture and checks the engine against the reference live rule at its last tick; returns the phase */
function checkEngine(fx: Fixture): string {
  const { engine, lastT } = replay(fx);
  const s = engine.snapshot();
  const want = expected(fx, lastT);
  if (want.grip) {
    expect(s.phase, fx.id).toBe('grip');
    expect(s.decision.status, fx.id).toBe('grip');
    return s.phase;
  }
  const nUsed = want.dec.final ? want.dec.nUsed : want.att.length;
  expect(s.attempts.map((a) => a.bout.index), fx.id).toEqual(want.att.slice(0, nUsed));
  expect([s.phase, s.decision.status, s.decision.nUsed, s.decision.percentRight, s.decision.answered], fx.id).toEqual([
    want.dec.final ? 'final' : 'watching',
    want.dec.status,
    want.dec.nUsed,
    want.dec.percentRight,
    want.dec.answered,
  ]);
  expect(Math.abs(s.decision.pRight - want.dec.pRight), fx.id).toBeLessThanOrEqual(1e-12);
  expect(s.decision.decidedMs, fx.id).toBe(want.dec.decidedMs);
  // every frozen attempt equals the reference's segment, its frozen time is at least decided + settle
  for (const a of s.attempts) {
    const r = fx.reference.segments[a.bout.index];
    expect([a.bout.onsetMs, a.bout.endMs, a.bout.decidedMs], fx.id).toEqual([r.onset_ms, r.end_ms, r.decided_ms]);
    // the prefix's median grid step differs from the full stream's by ulps (still_before / still_after features)
    expect(Math.abs(a.pApproach - r.proba.approach), fx.id).toBeLessThanOrEqual(1e-12);
    expect(a.frozenAtMs, fx.id).toBeGreaterThanOrEqual(r.freeze_after_ms);
    expect(Math.abs(a.outputs.m4.pRight - (r.models?.m4?.p ?? NaN)), fx.id).toBeLessThanOrEqual(1e-12);
  }
  if (s.attempts.length || s.ignored.length) expect(s.restUpZ, fx.id).toBe(fx.reference.grip.rest_up_z);
  expect(s.compare.map((c) => [c.modelId, c.k, c.percentRight]), fx.id).toEqual([['m4', s.decision.nUsed, s.decision.percentRight]]);
  return s.phase;
}

describe('live session engine (detect/engine.ts)', () => {
  it('freezes the attempts and reaches the decision of the reference live rule, on every synthetic fixture', () => {
    const phases = fixtures.map(checkEngine);
    expect(phases).toContain('final');
    expect(phases).toContain('grip');
  });

  it('gives the same final snapshot whatever the chunking of the samples', () => {
    const fx = fixtures[1];
    const a = replay(fx).engine.snapshot();
    const r = (seed: number) => {
      let x = seed;
      return () => ((x = (x * 1103515245 + 12345) % 2147483648), 1 + (x % 23));
    };
    for (const seed of [1, 2, 3]) {
      const next = r(seed);
      const b = replay(fx, () => next()).engine.snapshot();
      expect(b.attempts.map((x) => x.bout.index)).toEqual(a.attempts.map((x) => x.bout.index));
      expect(b.decision).toEqual(a.decision);
    }
  });

  it('ends at maxStreamMs as stream_end / no_attempt, ignores samples once final, and resets', () => {
    const fx = fixtures[0];
    const early = replay(fx, () => 1, 1500).engine; // before any bout is frozen
    expect(early.snapshot()).toMatchObject({ phase: 'final', decision: { status: 'no_attempt', answered: false, final: true } });
    const n = early.snapshot().nSamples;
    early.push(samplesOf(fx, 99999).slice(0, 10));
    expect(early.snapshot().nSamples).toBe(n);
    early.reset();
    expect(early.snapshot()).toMatchObject({ phase: 'watching', nSamples: 0, attempts: [], lastEvent: { type: 'reset' } });
    const later = replay(fx, () => 1, 4000).engine; // after the first attempt, before the decision
    expect(later.snapshot().decision).toMatchObject({ status: 'stream_end', nUsed: 1, answered: true, final: true });
  });

  it('setDriver recomputes from the frozen attempts; notifyGap resets only while watching; dispose stops', () => {
    const fx = fixtures[0];
    const { deps } = fakePipeline(streamOf(fx));
    const m4 = m4TestModel();
    const engine = makeCreateSessionEngine(deps)({
      normalization: { gravitySign: 1, rrVerdict: 'w3c', rrUnits: 'deg_per_s', engine: 'blink', quantizeIos: false, source: 'engine_prior' },
      models: [m4, meanPModel(m4)],
      driver: 'm4',
    });
    const xs = samplesOf(fx);
    engine.push(xs.slice(0, 260)); // first attempt frozen (decided 2650 ms + 1 s): m4's P_1 = 0.86, still watching
    let s = engine.snapshot();
    expect([s.phase, s.attempts.length, s.decision.modelId, s.decision.status]).toEqual(['watching', 1, 'm4', 'watching']);
    expect(s.compare.map((c) => [c.modelId, c.k])).toEqual([
      ['m4', 1],
      ['yaw', 1],
    ]);
    // the mean-P model is confident after the same attempt (P_1 = 0.90): switching ends the session at once
    engine.setDriver('yaw');
    s = engine.snapshot();
    expect([s.phase, s.decision.modelId, s.decision.status, s.decision.nUsed]).toEqual(['final', 'yaw', 'confident', 1]);
    // back to m4 on a final session: recomputed with the stream ended (one attempt, not confident -> stream_end)
    engine.setDriver('m4');
    expect(engine.snapshot().decision).toMatchObject({ modelId: 'm4', status: 'stream_end', nUsed: 1, answered: true, final: true });
    engine.notifyGap(); // a final answer stays
    expect(engine.snapshot().phase).toBe('final');
    engine.reset();
    engine.push(xs.slice(0, 100));
    expect(engine.snapshot()).toMatchObject({ phase: 'watching', driver: 'm4' });
    engine.notifyGap(); // watching: a new stream starts
    expect(engine.snapshot()).toMatchObject({ phase: 'watching', nSamples: 0, attempts: [], lastEvent: { type: 'gap' } });
    engine.push(xs);
    expect(engine.snapshot().phase).toBe('final');
    const seen: string[] = [];
    const off = engine.subscribe((x) => seen.push(x.phase));
    engine.dispose();
    off();
    expect(seen).toEqual(['stopped']);
    engine.push(xs);
    expect(engine.snapshot().phase).toBe('stopped');
  });
});

describe.skipIf(!realFixturesPresent())('live session engine on the local real fixtures (git-ignored, absent in CI)', () => {
  it('reaches the decision of the reference live rule on every real fixture', () => {
    const phases = loadFixtures(REAL_DIR).map(checkEngine);
    // eslint-disable-next-line no-console
    console.log(`engine replay, real: ${phases.length} fixtures, phases ${JSON.stringify(phases.reduce<Record<string, number>>((m, p) => ((m[p] = (m[p] ?? 0) + 1), m), {}))}`);
    expect(phases.length).toBeGreaterThan(0);
  }, 120000);
});
