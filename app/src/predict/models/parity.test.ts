/**
 * Parity of the TypeScript models with the Python reference (analysis/stage3/reference.py reference_predict, which
 * runs the Stage 2 functions themselves and checks its m4 view against m4_production.predict_stream).
 *
 * Every golden fixture holds, per detected segment with a window: the float32 (17, 64) window, the CNN body
 * channels, and every model's {available, z_raw, p, z} with its inputs (CNN member logits, PHYS6 features6, yaw
 * x8, m4 p_attempt); per model the k-curve P_k over the attempts in decision order and the stopping decision, both
 * offline (the whole recording) and live (the attempts the engine would have frozen). The models here read the
 * fixture's windows; the attempt list is taken from the fixture (choosing it is detect/'s job).
 *
 * Committed fixtures are synthetic (src/predict/__fixtures__). Real recordings stay in
 * analysis/data/stage3_fixtures (git-ignored): that suite is skipped when the folder is absent (CI). Real fixtures
 * are named by their position only, never by id.
 *
 * Tolerances (docs/stage3/PORT_SPEC.md 5.6): z_raw of the logistic models 1e-9; CNN z_raw and member logits 1e-6;
 * calibrated p, z and P_k 1e-6; body channels within one float32 ulp; availability, attempts, statuses, n_used,
 * answered and percent_right identical. The worst differences are printed at the end.
 */
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { afterAll, describe, expect, it } from 'vitest';
import type { ApproachWindow, ModelAttemptOutput, ModelId, StoppingParams } from '../types';
import { aggregateOutputs, isUsable } from './aggregate';
import { cnnForward } from './finalCnn';
import { m4Forward, phys6Forward, PHYS6_FEATURES, yawForward } from './logistic';
import { roundHalfEven } from './numeric';
import { MODEL_ORDER, bundle, loadModels, specs, stoppingParams } from './registry';

// ───────────────────────────── fixture format ('nosetouch-fixture-1') ─────────────────────────────

type Num = number | null;

interface FxModelOut {
  available: boolean;
  z_raw: Num;
  p: Num;
  z: Num;
  member_logits?: Num[];
  features6?: Num[] | null;
  x8?: Num[];
  p_attempt?: Num;
}

interface FxSegment {
  seg: number;
  onset_ms: number;
  end_ms: number;
  decided_ms: number;
  canonical_ok: boolean;
  is_attempt: boolean;
  window?: Num[][];
  body?: Num[][];
  models?: Record<ModelId, FxModelOut>;
}

interface FxDecision {
  p_right: Num;
  n_used: number;
  stopped: string;
  answered: boolean;
  final: boolean;
  decided_ms: Num;
  percent_right: number;
  k_curve: { k: number; p: number; seg: number; decided_ms: number }[];
  n_unavailable: number;
}

interface Fixture {
  format: string;
  id: string;
  kind: string;
  models: Record<ModelId, string>;
  reference: {
    t_last_ms: Num;
    attempts: number[];
    segments: FxSegment[];
    decisions: Record<ModelId, FxDecision>;
    grip: { grip_warning: boolean };
  };
  live: { attempts: number[]; decisions: Record<ModelId, FxDecision>; max_stream_ms: Num };
}

const nan = (v: Num | undefined): number => (v === null || v === undefined ? NaN : v);

function readFixture(url: URL): Fixture {
  return JSON.parse(gunzipSync(readFileSync(url)).toString('utf8')) as Fixture;
}

function windowOf(s: FxSegment): ApproachWindow {
  const rows = s.window!;
  const data = new Float32Array(17 * 64);
  rows.forEach((r, c) => r.forEach((v, t) => (data[c * 64 + t] = nan(v))));
  return { data, onsetMs: s.onset_ms, endMs: s.end_ms };
}

// ───────────────────────────── worst differences ─────────────────────────────

const worst = new Map<string, { max: number; n: number }>();
/** the suite being checked ('synthetic' or 'real'), so each reports its own worst differences */
let suite = 'synthetic';

function note(quantity: string, a: number, b: number): number {
  const d = Number.isNaN(a) && Number.isNaN(b) ? 0 : Math.abs(a - b);
  const key = `${suite}: ${quantity}`;
  const w = worst.get(key) ?? { max: 0, n: 0 };
  w.n += 1;
  if (!(d <= w.max)) w.max = d;
  worst.set(key, w);
  return d;
}

/** |a - b| in units of the float32 spacing at b (0 = identical float32 values) */
function f32ulps(a: number, b: number): number {
  if (a === b) return 0;
  const e = Math.max(Math.abs(b), 1.1754943508222875e-38);
  const ulp = Math.pow(2, Math.floor(Math.log2(e)) - 23);
  return Math.abs(a - b) / ulp;
}

const tally = new Map<string, { fixtures: number; windows: number; attempts: number; unavailable: number; decisions: number }>();

afterAll(() => {
  const counts = [...tally.entries()].map(
    ([s, c]) =>
      `${s}: ${c.fixtures} fixtures, ${c.windows} windows, ${c.attempts} attempts, ` +
      `${c.unavailable} unavailable model outputs, ${c.decisions} decisions (4 models x offline/live)`,
  );
  const lines = [...worst.entries()].sort().map(([k, w]) => `${k.padEnd(45)} max |d| = ${w.max.toExponential(2)}  (n = ${w.n})`);
  console.log(`models parity\n${counts.join('\n')}\nworst differences:\n${lines.join('\n')}`);
});

// ───────────────────────────── the page's stopping rule (reference.stopping) ─────────────────────────────

/** analysis/stage3/reference.py stopping(): lane m4's rule on one model's P_1..P_K (test-only copy) */
function stopping(pk: number[], decided: number[], st: StoppingParams, streamEnded: boolean) {
  let p = 0.5;
  let k = 0;
  for (k = 1; k <= Math.min(st.kMax, pk.length); k++) {
    p = pk[k - 1];
    if (Math.max(p, 1 - p) >= st.pStop) {
      return { p_right: p, n_used: k, stopped: 'confident', answered: true, final: true, decided_ms: decided[k - 1] };
    }
  }
  k = Math.min(st.kMax, pk.length);
  if (k === 0) {
    return { p_right: 0.5, n_used: 0, stopped: streamEnded ? 'no_attempt' : 'watching', answered: false, final: streamEnded, decided_ms: NaN };
  }
  if (k < st.kMax && !streamEnded) {
    return { p_right: p, n_used: k, stopped: 'watching', answered: false, final: false, decided_ms: decided[k - 1] };
  }
  return {
    p_right: p,
    n_used: k,
    stopped: k >= st.kMax ? 'k_max' : 'stream_end',
    answered: Math.max(p, 1 - p) >= st.pAnswer,
    final: true,
    decided_ms: decided[k - 1],
  };
}

// ───────────────────────────── the checks ─────────────────────────────

const TOL_LOGISTIC_ZRAW = 1e-9;
const TOL_CNN = 1e-6;
const TOL_P = 1e-6;

type Outputs = Map<number, Record<ModelId, ModelAttemptOutput>>;

/** every model on every windowed segment; compares each output and its inputs with the fixture */
function checkSegments(fx: Fixture): Outputs {
  const b = bundle();
  const outs: Outputs = new Map();
  for (const s of fx.reference.segments) {
    if (!s.window) {
      expect(s.canonical_ok).toBe(false);
      continue;
    }
    const w = windowOf(s);
    const ref = s.models!;
    const cnn = cnnForward(b.cnn, { window: w });
    const m4 = m4Forward(b.m4, w);
    const ph = phys6Forward(b.phys6, w);
    const yaw = yawForward(b.yaw, w);
    const got: Record<ModelId, ModelAttemptOutput> = { cnn: cnn.output, m4: m4.output, phys6: ph.output, yaw: yaw.output };
    outs.set(s.seg, got);
    for (const id of MODEL_ORDER) {
      const r = ref[id];
      const g = got[id];
      expect(g.available ?? true, `${id} available, seg ${s.seg}`).toBe(r.available);
      if (!r.available) {
        expect(Number.isNaN(g.pRight)).toBe(true);
        continue;
      }
      const tolZ = id === 'cnn' ? TOL_CNN : TOL_LOGISTIC_ZRAW;
      expect(note(`${id}.z_raw`, g.zRaw, nan(r.z_raw)), `${id} z_raw, seg ${s.seg}`).toBeLessThanOrEqual(tolZ);
      expect(note(`${id}.p`, g.pRight, nan(r.p)), `${id} p, seg ${s.seg}`).toBeLessThanOrEqual(TOL_P);
      expect(note(`${id}.z`, g.z, nan(r.z)), `${id} z, seg ${s.seg}`).toBeLessThanOrEqual(TOL_P);
    }
    // inputs and intermediates
    ref.cnn.member_logits?.forEach((v, i) => {
      expect(note('cnn.member_logit', cnn.memberLogits[i], nan(v))).toBeLessThanOrEqual(TOL_CNN);
    });
    if (s.body) {
      let maxUlps = 0;
      s.body.forEach((row, c) =>
        row.forEach((v, t) => {
          const u = f32ulps(cnn.body[c * 64 + t], nan(v));
          note('cnn.body (float32 values)', cnn.body[c * 64 + t], nan(v));
          maxUlps = Math.max(maxUlps, u);
        }),
      );
      const key = `${suite}: cnn.body (float32 ulps)`;
      const w2 = worst.get(key) ?? { max: 0, n: 0 };
      worst.set(key, { max: Math.max(w2.max, maxUlps), n: w2.n + 1 });
      expect(maxUlps, `body channels, seg ${s.seg}`).toBeLessThanOrEqual(1);
    }
    ref.yaw.x8?.forEach((v, i) => expect(yaw.x8[i]).toBe(nan(v)));
    // the yaw reference was fitted and evaluated in float32 arithmetic; the page computes in float64 (the fixtures'
    // z_raw and p): how far that is from the evaluated float32 path
    const pf32 = (ref.yaw as FxModelOut & { p_f32?: Num }).p_f32;
    if (ref.yaw.available && pf32 !== undefined && pf32 !== null) {
      expect(note('yaw.p vs the float32-evaluated p (info)', yaw.output.pRight, pf32)).toBeLessThanOrEqual(1e-6);
    }
    if (ref.m4.p_attempt !== undefined && ref.m4.p_attempt !== null) {
      expect(note('m4.p_attempt', m4.pAttempt, ref.m4.p_attempt)).toBeLessThanOrEqual(TOL_P);
    }
    const f6 = ref.phys6.features6;
    if (f6) {
      const v = ph.features.values;
      expect(v, `phys6 features, seg ${s.seg}`).not.toBeNull();
      PHYS6_FEATURES.forEach((name, i) => {
        const d = note('phys6.features6 (relative)', v![name] / (Math.abs(nan(f6[i])) || 1), nan(f6[i]) / (Math.abs(nan(f6[i])) || 1));
        expect(d, `${name}, seg ${s.seg}`).toBeLessThanOrEqual(1e-9);
      });
    }
  }
  return outs;
}

/** k-curves and decisions of every model, offline and live, from the TS per-attempt outputs */
function checkDecisions(fx: Fixture, outs: Outputs): void {
  const st = stoppingParams();
  const models = loadModels();
  const grip = fx.reference.grip.grip_warning;
  const modes = [
    { name: 'offline', attempts: fx.reference.attempts, decisions: fx.reference.decisions, streamEnded: true },
    {
      name: 'live',
      attempts: fx.live.attempts,
      decisions: fx.live.decisions,
      streamEnded: fx.live.max_stream_ms !== null && nan(fx.reference.t_last_ms) >= fx.live.max_stream_ms,
    },
  ];
  const segs = new Map(fx.reference.segments.map((s) => [s.seg, s]));
  for (const mode of modes) {
    for (const model of models) {
      const id = model.spec.id;
      const ref = mode.decisions[id];
      const usable = mode.attempts.filter((k) => isUsable(outs.get(k)![id]));
      expect(usable, `${mode.name} ${id}: usable attempts`).toEqual(ref.k_curve.map((r) => r.seg));
      expect(mode.attempts.length - usable.length).toBe(ref.n_unavailable);
      const o = usable.map((k) => outs.get(k)![id]);
      const pk = o.map((_, k) => model.aggregate(o.slice(0, k + 1)));
      pk.forEach((p, k) => {
        expect(note(`${id}.P_k`, p, ref.k_curve[k].p), `${mode.name} ${id} P_${k + 1}`).toBeLessThanOrEqual(TOL_P);
        // the registry's HandModel and the bare aggregation agree
        expect(aggregateOutputs(model.spec.aggregation, o.slice(0, k + 1))).toBe(p);
      });
      const decided = usable.map((k) => segs.get(k)!.decided_ms);
      let dec = stopping(pk, decided, st, mode.streamEnded);
      if (grip) dec = { p_right: 0.5, n_used: 0, stopped: 'grip', answered: false, final: true, decided_ms: NaN };
      const tag = `${mode.name} ${id}`;
      expect(dec.stopped, `${tag} status`).toBe(ref.stopped);
      expect(dec.n_used, `${tag} n_used`).toBe(ref.n_used);
      expect(dec.answered, `${tag} answered`).toBe(ref.answered);
      expect(dec.final, `${tag} final`).toBe(ref.final);
      expect(roundHalfEven(100 * dec.p_right), `${tag} percent`).toBe(ref.percent_right);
      expect(note(`decision.p_right`, dec.p_right, nan(ref.p_right))).toBeLessThanOrEqual(TOL_P);
      expect(note(`decision.decided_ms`, dec.decided_ms, nan(ref.decided_ms))).toBe(0);
    }
  }
}

function checkFixture(fx: Fixture): { windows: number; attempts: number; decisions: number } {
  expect(fx.format).toBe('nosetouch-fixture-1');
  // the fixture was computed with exactly the bundled model files
  for (const s of specs) expect(fx.models[s.id], `${s.id} model hash`).toBe(s.sha256);
  const outs = checkSegments(fx);
  checkDecisions(fx, outs);
  const t = tally.get(suite) ?? { fixtures: 0, windows: 0, attempts: 0, unavailable: 0, decisions: 0 };
  t.fixtures += 1;
  t.windows += outs.size;
  t.attempts += fx.reference.attempts.length;
  for (const o of outs.values()) for (const id of MODEL_ORDER) if (o[id].available === false) t.unavailable += 1;
  t.decisions += 2 * MODEL_ORDER.length;
  tally.set(suite, t);
  return { windows: outs.size, attempts: fx.reference.attempts.length, decisions: 2 * MODEL_ORDER.length };
}

// ───────────────────────────── suites ─────────────────────────────

const SYN = new URL('../__fixtures__/', import.meta.url);
const synIndex = JSON.parse(readFileSync(new URL('index.json', SYN), 'utf8')) as { cases: { file: string; id: string }[] };

describe('models parity: committed synthetic fixtures', () => {
  it('has the synthetic cases', () => {
    expect(synIndex.cases.length).toBeGreaterThanOrEqual(5);
  });
  it.each(synIndex.cases.map((c) => [c.id, c.file]))('%s', (_id, file) => {
    suite = 'synthetic';
    const fx = readFixture(new URL(file, SYN));
    expect(fx.kind).toBe('synthetic');
    const n = checkFixture(fx);
    expect(n.windows).toBeGreaterThan(0);
  });
});

const REAL = new URL('../../../../analysis/data/stage3_fixtures/', import.meta.url);
const realIndexUrl = new URL('index.json', REAL);
const hasReal = existsSync(realIndexUrl);

describe.skipIf(!hasReal)('models parity: local real fixtures (analysis/data/stage3_fixtures, never committed)', () => {
  const cases = hasReal ? (JSON.parse(readFileSync(realIndexUrl, 'utf8')) as { cases: { file: string }[] }).cases : [];
  it('has the real cases', () => {
    expect(cases.length).toBeGreaterThan(0);
  });
  it.each(cases.map((c, i) => [i + 1, c.file]))('real fixture #%i', (_i, file) => {
    suite = 'real';
    const fx = readFixture(new URL(file, REAL));
    checkFixture(fx);
  });
});
