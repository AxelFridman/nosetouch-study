/**
 * End to end with the real pipeline/ and models/: the raw motion of each fixture -> pipeline.streamFrom -> detect/
 * (bouts, windows cut by pipeline.canonical / timeNormalised, attempts, grip, stopping) -> every bundled model, against
 * analysis/stage3/reference.py (offline decisions of the four models and m4's predict_stream dict); and the live
 * engine fed the raw samples one by one through pipeline's LiveBuffer, against the reference's live rule.
 */
import { describe, expect, it } from 'vitest';
import type { DetectorBundle, LiveSample, ModelId, StreamNormalization } from '../types';
import { loadModels, detector as registryDetector } from '../models/registry';
import { createIncrementalStreamFrom, pipeline } from '../pipeline';
import { makeCreateSessionEngine } from './engine';
import { analyseStream, predictStream } from './predict';
import { stopOnCurve } from './stopping';
import { REAL_DIR, SYNTH_DIR, loadFixtures, numOf, realFixturesPresent, type Fixture } from './testSupport';

type Motion = Parameters<typeof pipeline.rawMotionFromRecord>[0];
interface RawFixture extends Fixture {
  motion: Motion;
  normalization: StreamNormalization;
}

const det: DetectorBundle = registryDetector();
const models = loadModels();
const MODELS: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];

function offline(fx: RawFixture, worst: Map<string, number>): void {
  const raw = pipeline.rawMotionFromRecord(fx.motion);
  const stream = pipeline.streamFrom(raw, fx.normalization);
  for (const m of models) {
    const id = m.spec.id;
    const want = fx.reference.decisions[id];
    const a = analyseStream(stream, m, det, pipeline, true);
    const d = a.decision;
    const where = `${fx.id} ${id}`;
    expect([d.status, d.nUsed, d.answered, d.percentRight], where).toEqual([want.stopped, want.n_used, want.answered, want.percent_right]);
    expect(d.decidedMs, where).toBe(want.decided_ms);
    const dp = Math.abs(d.pRight - want.p_right);
    worst.set(`P(right) ${id}`, Math.max(worst.get(`P(right) ${id}`) ?? 0, dp));
    expect(dp, where).toBeLessThanOrEqual(1e-6);
    // the model's own k-curve over the same attempts
    const kc = want.k_curve.map((e) => e.p).slice(0, det.stopping.kMax);
    expect(a.kCurve.length, where).toBe(kc.length);
    a.kCurve.forEach((p, k) => {
      const e = Math.abs(p - kc[k]);
      worst.set(`P_k ${id}`, Math.max(worst.get(`P_k ${id}`) ?? 0, e));
      expect(e, where).toBeLessThanOrEqual(1e-6);
    });
  }
  // m4: predict_stream's whole dict, from the raw motion
  const m4 = models.find((m) => m.spec.id === 'm4')!;
  const ps = predictStream(raw, fx.normalization, m4, det, pipeline);
  const w = fx.reference.m4_predict_stream;
  expect([ps.status, ps.nAttemptsUsed, ps.percentRight, ps.answered, ps.gripWarning], fx.id).toEqual([
    w.status,
    w.n_attempts_used,
    w.percent_right,
    w.answered,
    w.grip_warning,
  ]);
  expect(Math.abs(numOf(ps.restUpZ) - numOf(w.rest_up_z)) || 0, fx.id).toBeLessThanOrEqual(1e-12);
  expect(ps.attempts.map((x) => x.decidedMs), fx.id).toEqual(w.attempts.map((x) => x.decided_ms));
}

function live(fx: RawFixture): string {
  let lastT = NaN;
  // the production configuration: pipeline's incremental streamFrom, one per engine (it restarts on reset)
  const incremental = createIncrementalStreamFrom();
  const engine = makeCreateSessionEngine({
    detector: det,
    pipeline: {
      ...pipeline,
      streamFrom: (raw, norm) => {
        const s = incremental(raw, norm);
        lastT = s.t[s.t.length - 1];
        return s;
      },
    },
  })({ normalization: fx.normalization, models, driver: 'm4' });
  const m = fx.motion;
  const vec = (x: number | null, y: number | null, z: number | null): [number, number, number] | null =>
    x === null || y === null || z === null ? null : [x, y, z];
  for (let i = 0; i < m.t.length; i++) {
    const s: LiveSample = {
      t: m.t[i] + 5000.25,
      ts: null,
      acc: vec(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
      accg: vec(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
      rr: m.rr_alpha[i] === null || m.rr_beta[i] === null || m.rr_gamma[i] === null ? null : { alpha: m.rr_alpha[i]!, beta: m.rr_beta[i]!, gamma: m.rr_gamma[i]! },
      interval: 16,
    };
    engine.push(s);
  }
  const s = engine.snapshot();
  const ref = fx.reference;
  const frozenAt = (k: number) => ref.segments[k].ended && ref.segments[k].freeze_after_ms <= lastT;
  if (ref.grip.grip_warning && ref.segments.some((_, k) => frozenAt(k))) {
    expect(s.phase, fx.id).toBe('grip');
    return s.phase;
  }
  const att = ref.attempts.filter(frozenAt);
  const kc = ref.decisions.m4.k_curve.slice(0, att.length);
  const dec = stopOnCurve('m4', kc.map((e) => e.p), kc.map((e) => e.decided_ms), det.stopping, false);
  expect(s.attempts.map((a) => a.bout.index), fx.id).toEqual(att.slice(0, dec.final ? dec.nUsed : att.length));
  expect([s.decision.status, s.decision.nUsed, s.decision.percentRight], fx.id).toEqual([dec.status, dec.nUsed, dec.percentRight]);
  expect(Math.abs(s.decision.pRight - dec.pRight), fx.id).toBeLessThanOrEqual(1e-6);
  // every model's output of every frozen attempt equals the reference's
  for (const a of s.attempts) {
    for (const id of MODELS) {
      const r = ref.segments[a.bout.index].models?.[id];
      const o = a.outputs[id];
      if (!r) continue;
      const avail = o.available !== false && Number.isFinite(o.z);
      expect(avail, `${fx.id} ${id}`).toBe(r.available);
      if (r.available) expect(Math.abs(o.pRight - numOf(r.p)), `${fx.id} ${id}`).toBeLessThanOrEqual(1e-6);
    }
  }
  return s.phase;
}

function run(label: string, fixtures: RawFixture[]): void {
  const worst = new Map<string, number>();
  const phases: Record<string, number> = {};
  for (const fx of fixtures) {
    offline(fx, worst);
    const p = live(fx);
    phases[p] = (phases[p] ?? 0) + 1;
  }
  // eslint-disable-next-line no-console
  console.log(
    `detect + pipeline + models, ${label}: ${fixtures.length} fixtures, live phases ${JSON.stringify(phases)}\n` +
      [...worst.entries()].map(([k, v]) => `${k.padEnd(16)} ${v.toExponential(2)}`).join('\n'),
  );
}

describe('end to end with pipeline/ and models/', () => {
  it('synthetic fixtures: every model offline, m4 predict_stream, and the live engine from raw samples', () => {
    run('synthetic', loadFixtures(SYNTH_DIR) as RawFixture[]);
  }, 120000);

  it.skipIf(!realFixturesPresent())('local real fixtures (git-ignored, absent in CI)', () => {
    run('real', loadFixtures(REAL_DIR) as RawFixture[]);
  }, 300000);
});
