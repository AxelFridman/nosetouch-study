/**
 * detect/ against the Python reference (analysis/stage3/reference.py reference_predict, the golden fixtures):
 * activity, bouts, the sample-by-sample tracker, window bounds and crops, gravity metadata, windows, features,
 * P(class), the attempt rule, the live freeze rule, the grip check, every model's stopping decision (offline and
 * live) and, with a minimal m4 hand model, predict_stream's whole dict.
 *
 * Committed fixtures are synthetic (app/src/predict/__fixtures__). Real recordings live only in
 * analysis/data/stage3_fixtures/ (git-ignored); that block is skipped where the folder is absent (CI).
 * Tolerances: docs/stage3/PORT_SPEC.md section 5.6. The worst difference per quantity is printed.
 */
import { describe, expect, it } from 'vitest';
import type { DetectorBundle, ModelId, SegmentClass } from '../types';
import { activity } from './activity';
import { BoutTracker, CausalDetector, findBouts, isFrozen, medianStep } from './bouts';
import { detectWindowPipeline } from './canonical';
import { segmentProba } from './classifier';
import { ALL_FEATURES } from './features';
import { createDetectFunctions } from './index';
import { restDetails } from './grip';
import { analyseStream, predictFromStream } from './predict';
import { segmentRows } from './rows';
import { gripDecision, stopOnCurve } from './stopping';
import {
  REAL_DIR,
  SYNTH_DIR,
  f32Ulps,
  loadDetector,
  loadFixtures,
  m4TestModel,
  numOf,
  realFixturesPresent,
  streamOf,
  windowOf,
  type Fixture,
} from './testSupport';

const MODELS: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];
const CLASSES: readonly SegmentClass[] = ['approach', 'return', 'other'];

/** worst |difference| per quantity, over every fixture of a run */
class Worst {
  readonly v = new Map<string, number>();
  add(key: string, d: number): void {
    const x = Number.isNaN(d) ? Infinity : d;
    this.v.set(key, Math.max(this.v.get(key) ?? 0, x));
  }
  /** |a - b| with NaN == NaN */
  diff(key: string, a: number, b: number): number {
    const d = Number.isNaN(a) && Number.isNaN(b) ? 0 : Math.abs(a - b);
    this.add(key, d);
    return d;
  }
  table(): string {
    return [...this.v.entries()].map(([k, d]) => `${k.padEnd(28)} ${d.toExponential(2)}`).join('\n');
  }
}

const rel = (a: number, b: number): number => (Number.isNaN(a) && Number.isNaN(b) ? 0 : Math.abs(a - b) / Math.max(1, Math.abs(b)));

function checkFixture(fx: Fixture, det: DetectorBundle, w: Worst, counts: Record<string, number>): void {
  const ref = fx.reference;
  const id = fx.id;
  const stream = streamOf(fx);
  const t = stream.t;
  const n = t.length;
  const p = det.segParams;
  const coefs = { sos: det.activitySos, zi: det.activityZi };
  counts.fixtures++;

  // ---- activity ----
  const act = activity(t, stream.gyro, p, coefs);
  expect(act.length, id).toBe(ref.activity.length);
  for (let i = 0; i < n; i++) expect(w.diff('activity (deg/s)', act[i], ref.activity[i]), `${id} activity[${i}]`).toBeLessThanOrEqual(1e-9);

  // ---- bouts (batch) ----
  expect(det.causalDelayMs, id).toBe(ref.detector.causal_delay_ms);
  w.diff('dt (ms)', medianStep(t), ref.detector.dt_ms);
  const bouts = findBouts(t, act, p, det.causalDelayMs);
  expect(bouts.length, `${id} bouts`).toBe(ref.segments.length);
  bouts.forEach((b, k) => {
    const s = ref.segments[k];
    const where = `${id} seg ${k}`;
    expect([b.index, b.iOn, b.iEnd, b.ended, b.startedActive], where).toEqual([s.seg, s.i_on, s.i_end, s.ended, s.started_active]);
    expect(w.diff('onset/end/decided (ms)', b.onsetMs, s.onset_ms), where).toBeLessThanOrEqual(1e-9);
    expect(w.diff('onset/end/decided (ms)', b.endMs, s.end_ms), where).toBeLessThanOrEqual(1e-9);
    expect(w.diff('onset/end/decided (ms)', b.decidedMs, s.decided_ms), where).toBeLessThanOrEqual(1e-9);
    expect(w.diff('still before/after (s)', b.stillBeforeS, s.still_before_s), where).toBeLessThanOrEqual(1e-12);
    expect(w.diff('still before/after (s)', b.stillAfterS, s.still_after_s), where).toBeLessThanOrEqual(1e-12);
    expect(w.diff('peak_dps', b.peakDps, s.peak_dps), where).toBeLessThanOrEqual(1e-9);
    counts.bouts++;
  });

  // ---- the same bouts fed sample by sample ----
  const tracker = new BoutTracker(p, det.causalDelayMs, medianStep(t));
  const streamed = [];
  for (let i = 0; i < n; i++) {
    const out = tracker.push(t[i], act[i]);
    for (const b of out) expect(b.iEnd + tracker.constants.nLook - 1, `${id} emitted at its look-ahead`).toBe(i);
    streamed.push(...out);
  }
  streamed.push(...tracker.finish());
  expect(streamed, `${id} tracker = batch`).toEqual(bouts);
  const causal = new CausalDetector(p, coefs, det.causalDelayMs, medianStep(t));
  const fromGyro = [];
  for (let i = 0; i < n; i++) fromGyro.push(...causal.push(t[i], stream.gyro.data[3 * i], stream.gyro.data[3 * i + 1], stream.gyro.data[3 * i + 2]));
  fromGyro.push(...causal.finish());
  expect(fromGyro, `${id} CausalDetector = batch`).toEqual(bouts);

  // ---- the live freeze rule: feeding only settled samples emits exactly the frozen bouts ----
  const settle = fx.live.settle_ms;
  const tLast = t[n - 1];
  const nLook = ref.detector.n_look;
  const live = new BoutTracker(p, det.causalDelayMs, medianStep(t));
  const emitted = [];
  for (let i = 0; i < n && t[i] + settle <= tLast; i++) emitted.push(...live.push(t[i], act[i]));
  const frozen = bouts.filter((b) => isFrozen(b, nLook, n, tLast, settle));
  expect(emitted.map((b) => b.index), `${id} settled feed = frozen`).toEqual(frozen.map((b) => b.index));
  bouts.forEach((b, k) => expect(isFrozen(b, nLook, n, tLast, settle), `${id} live_frozen ${k}`).toBe(ref.segments[k].live_frozen));

  // ---- segment table: crop, gravity, window, features, P(class), attempt rule ----
  const rows = segmentRows(stream, det, detectWindowPipeline, act);
  rows.forEach((r, k) => {
    const s = ref.segments[k];
    const where = `${id} seg ${k}`;
    expect(w.diff('crop bounds (ms)', r.crop.startMs, s.crop.start_ms), where).toBe(0);
    expect(w.diff('crop bounds (ms)', r.crop.stopMs, s.crop.stop_ms), where).toBe(0);
    expect(r.canonicalOk, where).toBe(s.canonical_ok);
    if (r.canonical && s.gravity) {
      expect([r.canonical.n, r.canonical.iOn, r.canonical.iEnd], where).toEqual([s.crop.n, s.crop.i_on, s.crop.i_end]);
      expect(r.canonical.t[0], where).toBe(s.crop.t0_ms);
      expect([r.canonical.gravityMethod, r.canonical.restPreN, r.canonical.restPostN], where).toEqual([
        s.gravity.method,
        s.gravity.pre_n,
        s.gravity.post_n,
      ]);
      expect(w.diff('closure (deg)', r.canonical.closureDeg, numOf(s.gravity.closure_deg)), where).toBeLessThanOrEqual(1e-9);
      const gi = r.canonical.iOn;
      const g = r.canonical.g.data;
      w.diff('|g| (m/s^2)', Math.hypot(g[3 * gi], g[3 * gi + 1], g[3 * gi + 2]), numOf(s.gravity.magnitude));
    }
    if (r.window) {
      const want = windowOf(s);
      let worst = 0;
      for (let i = 0; i < want.length; i++) {
        const u = f32Ulps(r.window.data[i], want[i]);
        worst = Math.max(worst, u);
        if (u > 0) counts.windowValuesOff++;
        counts.windowValues++;
      }
      w.add('window (float32 ulps)', worst);
      expect(worst, `${where} window`).toBeLessThanOrEqual(1);
    }
    expect(Object.keys(r.features).sort(), where).toEqual([...ALL_FEATURES].sort());
    for (const name of det.segmentClassifier.features) {
      const a = r.features[name];
      const b = numOf(s.features[name]);
      const d = rel(a, b);
      w.add('features (rel)', d);
      expect(d, `${where} ${name}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-8);
    }
    for (const c of CLASSES) expect(w.diff('P(class)', r.proba[c], s.proba[c]), `${where} P(${c})`).toBeLessThanOrEqual(1e-9);
    // the classifier alone, from the reference's own features
    const fromRef = segmentProba(det.segmentClassifier, Object.fromEntries(Object.entries(s.features).map(([k, v]) => [k, numOf(v)])));
    for (const c of CLASSES) expect(w.diff('P(class) | ref features', fromRef[c], s.proba[c]), where).toBeLessThanOrEqual(1e-12);
    expect([r.cls, r.isAttempt], where).toEqual([s.cls, s.is_attempt]);
    if (r.isAttempt) counts.attempts++;
  });

  // ---- grip check ----
  const rest = restDetails(stream, det);
  const g = ref.grip;
  expect(w.diff('rest_up_z', rest.restUpZ, numOf(g.rest_up_z)), id).toBeLessThanOrEqual(1e-12);
  expect([rest.nSel, rest.fallback, rest.nDefaultBouts], id).toEqual([g.n_sel, g.fallback, g.n_default_bouts]);
  expect(w.diff('first DEFAULT onset (ms)', rest.firstOnsetMs === Infinity ? NaN : rest.firstOnsetMs, numOf(g.first_onset_ms)), id).toBeLessThanOrEqual(1e-9);

  // ---- every model's decision from its k-curve (offline and live) ----
  for (const [mode, block, ended] of [
    ['offline', ref.decisions, true],
    ['live', fx.live.decisions, false],
  ] as const) {
    for (const m of MODELS) {
      const d = block[m];
      const where = `${id} ${mode} ${m}`;
      const dec = g.grip_warning
        ? gripDecision(m)
        : stopOnCurve(m, d.k_curve.map((e) => e.p), d.k_curve.map((e) => e.decided_ms), det.stopping, ended);
      expect([dec.status, dec.nUsed, dec.answered, dec.final, dec.percentRight], where).toEqual([
        d.stopped,
        d.n_used,
        d.answered,
        d.final,
        d.percent_right,
      ]);
      expect(dec.pRight, where).toBe(d.p_right);
      expect(dec.decidedMs, where).toBe(d.decided_ms);
      counts.decisions++;
    }
  }

  // ---- attempts in decision order (offline) and the frozen ones (live) ----
  const order = rows
    .filter((r) => r.isAttempt)
    .sort((a, b) => a.bout.decidedMs - b.bout.decidedMs || a.bout.index - b.bout.index)
    .map((r) => r.bout.index);
  expect(order, `${id} attempts`).toEqual(ref.attempts);
  expect(order.filter((k) => isFrozen(bouts[k], nLook, n, tLast, settle)), `${id} live attempts`).toEqual(fx.live.attempts);

  // ---- predict_stream end to end with lane m4's hand model ----
  const m4 = m4TestModel();
  const ps = predictFromStream(stream, m4, det, detectWindowPipeline);
  const want = ref.m4_predict_stream;
  expect([ps.status, ps.answered, ps.nAttemptsUsed, ps.percentRight, ps.gripWarning], id).toEqual([
    want.status,
    want.answered,
    want.n_attempts_used,
    want.percent_right,
    want.grip_warning,
  ]);
  expect(w.diff('predict_stream P(right)', ps.pRight, want.p_right), id).toBeLessThanOrEqual(1e-6);
  expect(w.diff('predict_stream decided_ms', numOf(ps.decidedMs), numOf(want.decided_ms)), id).toBeLessThanOrEqual(1e-9);
  expect(w.diff('rest_up_z', numOf(ps.restUpZ), numOf(want.rest_up_z)), id).toBeLessThanOrEqual(1e-12);
  expect(ps.attempts.length, id).toBe(want.attempts.length);
  ps.attempts.forEach((a, k) => {
    const b = want.attempts[k];
    expect(w.diff('onset/end/decided (ms)', a.onsetMs, b.onset_ms), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('onset/end/decided (ms)', a.decidedMs, b.decided_ms), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('P(class)', a.pApproach, b.p_approach), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('m4 attempt P(right)', a.pRight, b.p_right), id).toBeLessThanOrEqual(1e-6);
  });
  expect(ps.segments.length, id).toBe(want.segments.length);
  ps.segments.forEach((s, k) => {
    const b = want.segments[k];
    expect(w.diff('P(class)', s.pApproach, b.p_approach), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('P(class)', s.pReturn, b.p_return), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('P(class)', s.pOther, b.p_other), id).toBeLessThanOrEqual(1e-9);
  });
  // m4's per-attempt outputs and its k-curve against the reference's
  const an = analyseStream(stream, m4, det, detectWindowPipeline);
  for (const { row, output } of an.attempts) {
    const o = ref.segments[row.bout.index].models?.m4;
    if (!o) continue;
    expect(w.diff('m4 z_raw', output.zRaw, numOf(o.z_raw)), id).toBeLessThanOrEqual(1e-9);
    expect(w.diff('m4 p', output.pRight, numOf(o.p)), id).toBeLessThanOrEqual(1e-6);
  }
  const kc = ref.decisions.m4.k_curve.map((e) => e.p);
  expect(an.kCurve.length, id).toBe(Math.min(kc.length, det.stopping.kMax));
  an.kCurve.forEach((pk, k) => expect(w.diff('m4 P_k', pk, kc[k]), id).toBeLessThanOrEqual(1e-6));
}

function run(label: string, fixtures: Fixture[]): void {
  const det = loadDetector();
  const w = new Worst();
  const counts = { fixtures: 0, bouts: 0, attempts: 0, decisions: 0, windowValues: 0, windowValuesOff: 0 };
  for (const fx of fixtures) checkFixture(fx, det, w, counts);
  // eslint-disable-next-line no-console
  console.log(`detect parity, ${label}: ${JSON.stringify(counts)}\n${w.table()}`);
}

describe('detect parity: committed synthetic fixtures', () => {
  it('matches reference_predict on every synthetic fixture', () => {
    const fixtures = loadFixtures(SYNTH_DIR);
    expect(fixtures.length).toBeGreaterThanOrEqual(5);
    run('synthetic', fixtures);
  });

  it('the DetectApi binding (createDetectFunctions) gives the same answers as the pure functions', () => {
    const det = loadDetector();
    const fx = loadFixtures(SYNTH_DIR)[1];
    const stream = streamOf(fx);
    // a stand-in for pipeline.streamFrom: the fixture's own standardised stream
    const api = createDetectFunctions({ ...detectWindowPipeline, streamFrom: () => stream }, det);
    const m4 = m4TestModel();
    const ps = api.predictStream({} as never, {} as never, m4, det);
    expect(ps).toEqual(predictFromStream(stream, m4, det, detectWindowPipeline));
    expect(ps.percentRight).toBe(fx.reference.m4_predict_stream.percent_right);
    expect(api.findBouts(stream, det.segParams).map((b) => b.iOn)).toEqual(fx.reference.segments.map((s) => s.i_on));
    expect(api.segmentRows(stream, det).map((r) => r.isAttempt)).toEqual(fx.reference.segments.map((s) => s.is_attempt));
    expect(api.restUpZ(stream, det)).toBe(fx.reference.grip.rest_up_z);
    expect(Array.from(api.activity(stream, det.grip.restParams))).toEqual(fx.reference.activity);
    expect(() => api.activity(stream, { ...det.segParams, smooth_hz: 3 })).toThrow();
  });
});

describe.skipIf(!realFixturesPresent())('detect parity: local real fixtures (analysis/data/stage3_fixtures, git-ignored)', () => {
  it('matches reference_predict on every real fixture', () => {
    run('real', loadFixtures(REAL_DIR));
  }, 120000);
});
