/**
 * END-TO-END PARITY of the page's engine with the Python reference (analysis/stage3/reference.py, the golden fixtures).
 *
 * Every fixture is replayed exactly as the live page feeds it: a LiveSource hands the recorded DeviceMotion samples, one
 * at a time and in time order, on a performance.now()-like clock, to createLiveSession (gap check, frame-throttled
 * snapshots), which pushes them into the engine that live/engine.ts `binding` creates (LiveBuffer -> incremental
 * streamFrom -> lane m4's detector -> the four bundled models -> each model's aggregation -> the stopping rule), with
 * the production normalisation of the fixture's browser engine. Once per driver model (cnn, m4, phys6, yaw).
 *
 * The engine's own deps (engineDeps(), the binding's) are wrapped only to read what they return: the LiveBuffer, the
 * stream of every tick and every window. Compared with the reference:
 *   stream     the whole live buffer through streamFrom, and the last tick's stream: t, f, acc, gyro, valid
 *   bouts      every frozen bout (attempts and ignored movements): bounds, indices, decision time, stillness, peak
 *   windows    the (17, 64) float32 window of every attempt (ulps)
 *   models     every model's z_raw, P(right) and z for every attempt, and its availability
 *   aggregates the driver's P_1..P_k and every compare row (each model's own k-aggregation after the same attempts)
 *   decisions  status, n_used, P(right), percent, answered, final, decided_ms; grip
 * The expected decision is the reference's live rule at the engine's last tick (attempts frozen by then: ended, look-ahead
 * complete, decided_ms + 1 s <= the tick's last grid time). A final decision is also checked against the fixture's own
 * live decision (Python) directly.
 *
 * Committed synthetic fixtures always run; the real ones (analysis/data/stage3_fixtures, git-ignored) only where they
 * exist. Real fixtures are named by position only in the output.
 */
import { describe, expect, it } from 'vitest';
import { makeCreateSessionEngine, type EngineDeps } from './detect/engine';
import { stopOnCurve } from './detect/stopping';
import { REAL_DIR, SYNTH_DIR, f32Ulps, loadFixtures, numOf, realFixturesPresent, windowOf, type Fixture } from './detect/testSupport';
import { binding, engineDeps, type EngineBinding } from './live/engine';
import { createLiveSession } from './live/session';
import type { ApproachWindow, EngineSnapshot, LiveBuffer, LiveSample, LiveSource, ModelId, Rows3, Stream, StreamNormalization } from './types';

type Motion = {
  t: number[];
  acc_x: (number | null)[];
  acc_y: (number | null)[];
  acc_z: (number | null)[];
  accg_x: (number | null)[];
  accg_y: (number | null)[];
  accg_z: (number | null)[];
  rr_alpha: (number | null)[];
  rr_beta: (number | null)[];
  rr_gamma: (number | null)[];
};
interface RawFixture extends Fixture {
  motion: Motion;
  normalization: StreamNormalization;
}

const MODELS: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];
/** a performance.now()-like origin: the engine only sees times relative to its first sample */
const CLOCK0 = 48213.37;
const TOL_P = 1e-6;

/** worst |difference| per quantity, and where */
class Worst {
  readonly v = new Map<string, { d: number; where: string }>();
  note(key: string, d: number, where: string): void {
    const e = this.v.get(key);
    if (!e || d > e.d || (Number.isNaN(d) && !Number.isNaN(e.d))) this.v.set(key, { d, where });
  }
  table(): string {
    return [...this.v.entries()].map(([k, e]) => `  ${k.padEnd(28)} ${e.d === 0 ? '0' : e.d.toExponential(2)}${e.d ? `  (${e.where})` : ''}`).join('\n');
  }
}

interface Probe {
  buffer: LiveBuffer | null;
  lastStream: Stream | null;
  windows: ApproachWindow[];
  ticks: number;
}

/** the binding's own deps, wrapped to read (never change) what they return */
function probed(probe: Probe): EngineBinding {
  const deps = engineDeps();
  const p = deps.pipeline;
  const wrapped: EngineDeps = {
    detector: deps.detector,
    pipeline: {
      ...p,
      createLiveBuffer: () => (probe.buffer = p.createLiveBuffer()),
      streamFrom: (raw, norm) => {
        const s = p.streamFrom(raw, norm);
        probe.lastStream = s;
        probe.ticks += 1;
        return s;
      },
      timeNormalised: (can, a, b) => {
        const w = p.timeNormalised(can, a, b);
        probe.windows.push(w);
        return w;
      },
    },
  };
  return { ...binding, createSessionEngine: (o) => makeCreateSessionEngine(wrapped)(o) };
}

function samplesOf(m: Motion): LiveSample[] {
  const v = (x: number | null, y: number | null, z: number | null): [number, number, number] | null =>
    x === null || y === null || z === null ? null : [x, y, z];
  return m.t.map((t, i) => ({
    t: CLOCK0 + t,
    ts: CLOCK0 + t,
    acc: v(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
    accg: v(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
    rr:
      m.rr_alpha[i] === null || m.rr_beta[i] === null || m.rr_gamma[i] === null
        ? null
        : { alpha: m.rr_alpha[i] as number, beta: m.rr_beta[i] as number, gamma: m.rr_gamma[i] as number },
    interval: 16,
  }));
}

/** a LiveSource that replays the samples synchronously, in time order, like devicemotion events */
function replaySource(engine: LiveSource['engine'], samples: readonly LiveSample[]): LiveSource {
  return {
    kind: 'browser',
    engine,
    start(onSample) {
      for (const s of samples) onSample(s);
      return Promise.resolve();
    },
    stop() {},
  };
}

function maxAbsRows(a: Rows3, b: (number | null)[][], n: number): number {
  let d = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < 3; j++) {
      const x = a.data[3 * i + j];
      const y = numOf(b[i]?.[j] ?? null);
      if (Number.isNaN(x) !== Number.isNaN(y)) return Infinity;
      if (!Number.isNaN(x)) d = Math.max(d, Math.abs(x - y));
    }
  return d;
}

/** the stream against the reference on its first n rows (t, f, acc, gyro; valid identical) */
function streamDiff(s: Stream, ref: Fixture['reference']['stream'], n: number): number {
  let d = 0;
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(s.t[i] - ref.t[i]));
  for (let i = 0; i < n; i++) if ((s.valid[i] === 1) !== ref.valid[i]) return Infinity;
  return Math.max(d, maxAbsRows(s.f, ref.f, n), maxAbsRows(s.acc, ref.acc, n), maxAbsRows(s.gyro, ref.gyro, n));
}

interface Tally {
  runs: number;
  phases: Record<string, number>;
  attempts: number;
  ignored: number;
  windows: number;
  outputs: number;
  finalVsPython: number;
}

async function replay(fx: RawFixture, driver: ModelId, label: string, worst: Worst, tally: Tally): Promise<EngineSnapshot> {
  const probe: Probe = { buffer: null, lastStream: null, windows: [], ticks: 0 };
  const bind = probed(probe);
  const engineName = fx.engine as LiveSource['engine'];
  // the production normalisation (engine prior) is what the fixture was computed with
  const norm = bind.normalization(engineName);
  expect({ ...norm }, label).toMatchObject({ ...fx.normalization, source: 'engine_prior' });

  const samples = samplesOf(fx.motion);
  const live = createLiveSession({
    binding: bind,
    driver,
    makeSource: () => replaySource(engineName, samples),
    restartSourceOnReset: false,
    onSnapshot: () => undefined,
    onFailure: (r) => {
      throw new Error(`${label}: source failure ${r}`);
    },
    schedule: (fn) => fn(),
    watchVisibility: false,
  });
  await live.start();
  const snap = live.engine.snapshot();
  const ref = fx.reference;
  const det = engineDeps().detector;
  const where = `${label} ${driver}`;
  tally.runs += 1;
  tally.phases[snap.phase] = (tally.phases[snap.phase] ?? 0) + 1;

  // ── stream: the live buffer (sample by sample, the recorder's rounding) through streamFrom = the reference stream
  expect(probe.buffer, where).not.toBeNull();
  const last = probe.lastStream!;
  const lastT = last.t[last.t.length - 1];
  if (driver === 'cnn') {
    // the engine stops reading once final: the stream of the buffer it holds equals the reference on that prefix,
    // except for the newest second, which the zero-phase low-pass may still change (nothing frozen reads it)
    const full = engineDeps().pipeline.streamFrom(probe.buffer!.raw(), norm);
    const nBuf = full.t.length;
    const settled = Math.max(0, nBuf - (snap.phase === 'watching' ? 0 : 60));
    expect(nBuf, where).toBeLessThanOrEqual(ref.n_grid);
    if (nBuf === ref.n_grid) worst.note('stream (whole buffer)', streamDiff(full, ref.stream, nBuf), label);
    else worst.note('stream (buffer, settled rows)', streamDiff(full, ref.stream, settled), label);
    const nTick = last.t.length;
    worst.note('stream (last tick, settled rows)', streamDiff(last, ref.stream, Math.max(0, nTick - 60)), label);
  }

  // ── the reference live rule at the engine's last tick
  const frozenAt = (k: number): boolean => ref.segments[k].ended && ref.segments[k].freeze_after_ms <= lastT;
  const decOf = (id: ModelId) => {
    const kc = ref.decisions[id].k_curve.filter((e) => frozenAt(e.seg));
    return stopOnCurve(id, kc.map((e) => e.p), kc.map((e) => e.decided_ms), det.stopping, false);
  };

  if (ref.grip.grip_warning && ref.segments.some((_, k) => frozenAt(k))) {
    expect(snap.phase, where).toBe('grip');
    // the fixture's own live view (Python: grip once a bout is frozen, as the engine applies it) says the same
    expect(fx.live.decisions[driver].stopped, where).toBe('grip');
    expect([snap.decision.status, snap.decision.answered, snap.decision.final], where).toEqual(['grip', false, true]);
    expect(snap.attempts.length, where).toBe(0);
    worst.note('rest_up_z (grip)', Math.abs(numOf(snap.restUpZ) - numOf(ref.grip.rest_up_z)), label);
    return snap;
  }
  expect(snap.phase, where).not.toBe('grip');

  const want = decOf(driver);
  // ── bouts: everything frozen up to the deciding attempt (a final engine reads no further)
  const att = ref.attempts.filter(frozenAt);
  const usedAtt = want.final ? att.slice(0, want.nUsed) : att;
  const lastSeg = want.final && usedAtt.length ? usedAtt[usedAtt.length - 1] : Infinity;
  const wantFrozen = ref.segments.filter((s, k) => frozenAt(k) && s.seg <= lastSeg).map((s) => s.seg);
  const gotFrozen = [...snap.attempts.map((a) => a.bout), ...snap.ignored.map((m) => m.bout)].sort((a, b) => a.index - b.index);
  expect(gotFrozen.map((b) => b.index), where).toEqual(wantFrozen);
  expect(snap.attempts.map((a) => a.bout.index), where).toEqual(usedAtt);
  for (const b of gotFrozen) {
    const r = ref.segments[b.index];
    const d = Math.max(
      Math.abs(b.onsetMs - r.onset_ms),
      Math.abs(b.endMs - r.end_ms),
      Math.abs(b.decidedMs - r.decided_ms),
      Math.abs(b.stillBeforeS - r.still_before_s),
      Math.abs(b.stillAfterS - r.still_after_s),
      Math.abs(b.peakDps - r.peak_dps),
    );
    worst.note('bout times/stillness/peak', d, label);
    expect([b.iOn, b.iEnd, b.ended, b.startedActive], where).toEqual([r.i_on, r.i_end, r.ended, r.started_active]);
    expect(d, where).toBeLessThanOrEqual(1e-9);
  }
  for (const m of snap.ignored) expect(ref.segments[m.bout.index].is_attempt, where).toBe(false);
  tally.ignored += snap.ignored.length;

  // ── windows and every model's output of every attempt
  for (const a of snap.attempts) {
    const r = ref.segments[a.bout.index];
    const w = probe.windows.find((x) => x.onsetMs === a.bout.onsetMs && x.endMs === a.bout.endMs);
    expect(w, where).toBeDefined();
    const rw = windowOf(r);
    let ulps = 0;
    for (let i = 0; i < rw.length; i++) ulps = Math.max(ulps, f32Ulps(w!.data[i], rw[i]));
    worst.note('window (float32 ulps)', ulps, label);
    expect(ulps, where).toBe(0);
    worst.note('P(approach)', Math.abs(a.pApproach - r.proba.approach), label);
    tally.windows += 1;
    for (const id of MODELS) {
      const o = a.outputs[id];
      const ro = r.models?.[id];
      expect(ro, `${where} ${id}`).toBeDefined();
      const avail = o.available !== false && Number.isFinite(o.z);
      expect(avail, `${where} ${id}`).toBe(ro!.available);
      if (!ro!.available) continue;
      const dz = Math.abs(o.zRaw - numOf(ro!.z_raw ?? null));
      const dp = Math.abs(o.pRight - numOf(ro!.p ?? null));
      const dzc = Math.abs(o.z - numOf(ro!.z ?? null));
      worst.note(`${id} z_raw`, dz, label);
      worst.note(`${id} P(right) per attempt`, dp, label);
      worst.note(`${id} z (calibrated)`, dzc, label);
      expect(dp, `${where} ${id}`).toBeLessThanOrEqual(TOL_P);
      expect(dz, `${where} ${id}`).toBeLessThanOrEqual(1e-6);
      tally.outputs += 1;
    }
  }
  tally.attempts += snap.attempts.length;

  // ── the driver's decision and its P_1..P_k
  const d = snap.decision;
  expect([d.modelId, d.status, d.nUsed, d.percentRight, d.answered, d.final, d.hand], where).toEqual([
    driver,
    want.status,
    want.nUsed,
    want.percentRight,
    want.answered,
    want.final,
    want.hand,
  ]);
  expect(d.decidedMs, where).toBe(want.decidedMs);
  worst.note('decision P(right)', Math.abs(d.pRight - want.pRight), label);
  expect(Math.abs(d.pRight - want.pRight), where).toBeLessThanOrEqual(TOL_P);
  expect(d.trajectory.length, where).toBe(want.trajectory.length);
  d.trajectory.forEach((p, k) => worst.note('P_k (driver)', Math.abs(p - want.trajectory[k]), label));
  expect(snap.phase, where).toBe(want.final ? 'final' : 'watching');
  if (d.final) {
    // a final decision never changes with more data: it is the fixture's own live decision (Python)
    const py = fx.live.decisions[driver];
    expect([d.status, d.nUsed, d.percentRight, d.answered], where).toEqual([py.stopped, py.n_used, py.percent_right, py.answered]);
    expect(d.decidedMs, where).toBe(py.decided_ms);
    worst.note('final P(right) vs Python live', Math.abs(d.pRight - py.p_right), label);
    tally.finalVsPython += 1;
  }

  // ── compare-all: every model's own aggregation after the attempts the driver used
  const usedSegs = new Set(want.final || want.nUsed ? usedAtt.slice(0, want.nUsed) : []);
  for (const c of snap.compare) {
    const kc = ref.decisions[c.modelId].k_curve.filter((e) => usedSegs.has(e.seg)).slice(0, det.stopping.kMax);
    const p = kc.length ? kc[kc.length - 1].p : 0.5;
    expect(c.k, `${where} compare ${c.modelId}`).toBe(kc.length);
    worst.note('compare-all P(right)', Math.abs(c.pRight - p), label);
    expect(Math.abs(c.pRight - p), `${where} compare ${c.modelId}`).toBeLessThanOrEqual(TOL_P);
  }
  live.dispose();
  return snap;
}

async function run(label: string, fixtures: RawFixture[]): Promise<void> {
  const worst = new Worst();
  const tally: Tally = { runs: 0, phases: {}, attempts: 0, ignored: 0, windows: 0, outputs: 0, finalVsPython: 0 };
  const answers: string[] = [];
  for (const [i, fx] of fixtures.entries()) {
    const name = label === 'real' ? `real #${String(i + 1).padStart(2, '0')}` : fx.id;
    const cells: string[] = [];
    for (const driver of MODELS) {
      const s = await replay(fx, driver, name, worst, tally);
      cells.push(`${driver} ${s.decision.status}${s.decision.final ? ` ${s.decision.percentRight}%` : ''}`);
    }
    if (label !== 'real') answers.push(`  ${name.padEnd(34)} ${cells.join(' | ')}`);
  }
  // eslint-disable-next-line no-console
  console.log(
    `END-TO-END PARITY (${label}): ${fixtures.length} fixtures x 4 drivers = ${tally.runs} live replays; phases ${JSON.stringify(tally.phases)}; ` +
      `${tally.attempts} attempts (${tally.windows} windows, ${tally.outputs} model outputs), ${tally.ignored} ignored movements, ` +
      `${tally.finalVsPython} final decisions equal to Python's live decision\n` +
      (answers.length ? `${answers.join('\n')}\n` : '') +
      `worst |difference| (TS live engine vs Python reference):\n${worst.table()}`,
  );
  for (const [k, e] of worst.v) {
    if (k.startsWith('window')) expect(e.d, k).toBe(0);
    else expect(e.d, `${k} at ${e.where}`).toBeLessThanOrEqual(k.startsWith('stream') ? 1e-9 : TOL_P);
  }
}

describe('end-to-end parity: the live engine of the page against the Python reference', () => {
  it('committed synthetic fixtures, every driver model', async () => {
    await run('synthetic', loadFixtures(SYNTH_DIR) as RawFixture[]);
  }, 300_000);

  it.skipIf(!realFixturesPresent())('local real fixtures (git-ignored, absent in CI), every driver model', async () => {
    await run('real', loadFixtures(REAL_DIR) as RawFixture[]);
  }, 600_000);
});
