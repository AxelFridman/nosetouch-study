/**
 * The two cost measures of the live engine change no number:
 *  - LiveDetectCache (detect/live.ts): the activity and the median grid step carried over from the previous tick
 *    equal activity() and medianStep() of the whole stream bit for bit, on every tick of a real incremental stream
 *    (the newest rows change at every tick: the zero-phase low-pass), gaps and repeats included;
 *  - deferScoringMs: running an attempt's models once the phone is still gives the same attempts, outputs and
 *    decision as running them at the freeze tick (the window is cut at the freeze tick either way).
 * Committed synthetic fixtures (CI) and the local real ones when present.
 */
import { describe, expect, it } from 'vitest';
import { activity } from '../detect/activity';
import { medianStep } from '../detect/bouts';
import { makeCreateSessionEngine } from '../detect/engine';
import { LiveDetectCache } from '../detect/live';
import { REAL_DIR, SYNTH_DIR, loadFixtures, realFixturesPresent, type Fixture } from '../detect/testSupport';
import type { EngineSnapshot, LiveSample, ModelId, Stream } from '../types';
import { binding, engineDeps } from './engine';

/** the recorder's motion columns of a fixture (milliseconds from the first sample; null = missing) */
type Motion = Record<'acc_x' | 'acc_y' | 'acc_z' | 'accg_x' | 'accg_y' | 'accg_z' | 'rr_alpha' | 'rr_beta' | 'rr_gamma', (number | null)[]> & {
  t: number[];
};
const motionOf = (fx: Fixture): Motion => (fx as unknown as { motion: Motion }).motion;

function samplesOf(m: Motion): LiveSample[] {
  const v = (x: number | null, y: number | null, z: number | null): [number, number, number] | null =>
    x === null || y === null || z === null ? null : [x, y, z];
  return m.t.map((t, i) => ({
    t: 1000 + t,
    ts: null,
    acc: v(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
    accg: v(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
    rr:
      m.rr_alpha[i] === null || m.rr_beta[i] === null || m.rr_gamma[i] === null
        ? null
        : { alpha: m.rr_alpha[i] as number, beta: m.rr_beta[i] as number, gamma: m.rr_gamma[i] as number },
    interval: 16,
  }));
}

/** 3 s more of the last sample, held still (lets a deferred attempt be scored before the end) */
function withStillTail(s: readonly LiveSample[]): LiveSample[] {
  const last = s[s.length - 1];
  const tail: LiveSample[] = [];
  for (let k = 1; k <= 180; k++) tail.push({ ...last, t: last.t + (k * 1000) / 60, rr: { alpha: 0, beta: 0, gamma: 0 } });
  return [...s, ...tail];
}

function fixtures(): Fixture[] {
  return [...loadFixtures(SYNTH_DIR), ...(realFixturesPresent() ? loadFixtures(REAL_DIR).slice(0, 12) : [])];
}

describe('LiveDetectCache = activity() and medianStep() of the whole stream, every tick', () => {
  it('bit for bit on the incremental stream of every fixture', () => {
    let ticks = 0;
    let resumed = 0;
    for (const fx of fixtures()) {
      const deps = engineDeps();
      const det = deps.detector;
      const norm = binding.normalization(fx.engine as never);
      const buf = deps.pipeline.createLiveBuffer();
      const cache = new LiveDetectCache(det);
      const samples = samplesOf(motionOf(fx));
      for (let i = 0; i < samples.length; i++) {
        buf.push(samples[i]);
        if (i % 6 !== 5 && i !== samples.length - 1) continue; // about one tick per 100 ms
        let stream: Stream;
        try {
          stream = deps.pipeline.streamFrom(buf.raw(), norm);
        } catch {
          continue;
        }
        const got = cache.update(stream);
        const want = activity(stream.t, stream.gyro, det.segParams, { sos: det.activitySos, zi: det.activityZi });
        expect(got.act.length, fx.id).toBe(want.length);
        for (let r = 0; r < want.length; r++) if (!Object.is(got.act[r], want[r])) expect([fx.id, r, got.act[r]]).toEqual([fx.id, r, want[r]]);
        expect(Object.is(got.dtMs, medianStep(stream.t)), fx.id).toBe(true);
        ticks += 1;
        if (want.length > 60) resumed += 1;
      }
    }
    expect(ticks).toBeGreaterThan(100);
    expect(resumed).toBeGreaterThan(50);
  }, 300_000);
});

describe('deferScoringMs changes when an attempt is scored, never its numbers', () => {
  const run = (fx: Fixture, driver: ModelId, deferScoringMs: number): EngineSnapshot => {
    const engine = makeCreateSessionEngine(engineDeps())({
      normalization: binding.normalization(fx.engine as never),
      models: binding.models(),
      driver,
      deferScoringMs,
    });
    for (const s of withStillTail(samplesOf(motionOf(fx)))) engine.push(s);
    return engine.snapshot();
  };

  it('same attempts, outputs and decision as scoring at the freeze tick', () => {
    let attempts = 0;
    for (const fx of fixtures()) {
      for (const driver of ['cnn', 'm4'] as const) {
        const now = run(fx, driver, 0);
        const later = run(fx, driver, 1500);
        const where = `${fx.id} ${driver}`;
        expect(later.phase, where).toBe(now.phase);
        expect(later.attempts.map((a) => a.bout.index), where).toEqual(now.attempts.map((a) => a.bout.index));
        later.attempts.forEach((a, k) => {
          for (const id of ['cnn', 'm4', 'phys6', 'yaw'] as const) {
            expect(Object.is(a.outputs[id].pRight, now.attempts[k].outputs[id].pRight), `${where} ${id}`).toBe(true);
          }
          expect(a.frozenAtMs, where).toBeGreaterThanOrEqual(now.attempts[k].frozenAtMs);
        });
        const d = later.decision;
        const e = now.decision;
        expect([d.status, d.nUsed, d.final, d.answered, d.percentRight, d.decidedMs], where).toEqual([
          e.status,
          e.nUsed,
          e.final,
          e.answered,
          e.percentRight,
          e.decidedMs,
        ]);
        expect(Object.is(d.pRight, e.pRight), where).toBe(true);
        attempts += now.attempts.length;
      }
    }
    expect(attempts).toBeGreaterThan(10);
  }, 300_000);
});
