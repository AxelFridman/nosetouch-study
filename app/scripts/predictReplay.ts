/**
 * Replays a predict-mode golden fixture through the page's engine in Node, exactly as the live page feeds it: the
 * recorded DeviceMotion samples, one at a time and in time order, into createLiveSession with the wired binding
 * (live/engine.ts) and the production normalisation of the fixture's browser engine. Prints one JSON line per driver
 * model. e2e/predict.spec.ts compares the page's P(right) with this (the browser must show Node's numbers).
 *
 *   npx tsx scripts/predictReplay.ts src/predict/__fixtures__/syn01_android_right_3touch.json.gz [cnn|m4|phys6|yaw]
 *
 * Fixtures: the committed synthetic ones (src/predict/__fixtures__) or the local real ones
 * (analysis/data/stage3_fixtures, git-ignored: never commit what this prints for them).
 */
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { binding } from '../src/predict/live/engine';
import { createLiveSession } from '../src/predict/live/session';
import type { BrowserEngine, LiveSample, ModelId } from '../src/predict/types';

type Col = (number | null)[];
interface FixtureMotion {
  t: number[];
  acc_x: Col;
  acc_y: Col;
  acc_z: Col;
  accg_x: Col;
  accg_y: Col;
  accg_z: Col;
  rr_alpha: Col;
  rr_beta: Col;
  rr_gamma: Col;
}

const MODELS: readonly ModelId[] = ['cnn', 'm4', 'phys6', 'yaw'];

export function samplesOf(m: FixtureMotion, clock0 = 48213.37): LiveSample[] {
  const v = (x: number | null, y: number | null, z: number | null): [number, number, number] | null =>
    x === null || y === null || z === null ? null : [x, y, z];
  return m.t.map((t, i) => ({
    t: clock0 + t,
    ts: clock0 + t,
    acc: v(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
    accg: v(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
    rr:
      m.rr_alpha[i] === null || m.rr_beta[i] === null || m.rr_gamma[i] === null
        ? null
        : { alpha: m.rr_alpha[i] as number, beta: m.rr_beta[i] as number, gamma: m.rr_gamma[i] as number },
    interval: 16,
  }));
}

async function replay(engine: BrowserEngine, samples: readonly LiveSample[], driver: ModelId) {
  const live = createLiveSession({
    binding,
    driver,
    makeSource: () => ({
      kind: 'browser',
      engine,
      start(onSample) {
        for (const s of samples) onSample(s);
        return Promise.resolve();
      },
      stop() {},
    }),
    restartSourceOnReset: false,
    onSnapshot: () => undefined,
    onFailure: (r) => {
      throw new Error(`source failure ${r}`);
    },
    schedule: (fn) => fn(),
    watchVisibility: false,
  });
  await live.start();
  const s = live.engine.snapshot();
  live.dispose();
  return {
    driver,
    phase: s.phase,
    status: s.decision.status,
    final: s.decision.final,
    answered: s.decision.answered,
    pRight: s.decision.pRight,
    percentRight: s.decision.percentRight,
    nUsed: s.decision.nUsed,
    attempts: s.attempts.length,
    compare: Object.fromEntries(s.compare.map((c) => [c.modelId, { pRight: c.pRight, percentRight: c.percentRight, k: c.k }])),
  };
}

async function main(): Promise<void> {
  const [file, only] = process.argv.slice(2);
  if (!file) throw new Error('usage: tsx scripts/predictReplay.ts <fixture.json.gz> [model]');
  const fx = JSON.parse(gunzipSync(readFileSync(file)).toString('utf8')) as { engine: BrowserEngine; motion: FixtureMotion };
  const samples = samplesOf(fx.motion);
  for (const driver of only ? [only as ModelId] : MODELS) process.stdout.write(`${JSON.stringify(await replay(fx.engine, samples, driver))}\n`);
}

void main().catch((e: unknown) => {
  process.stderr.write(`${e instanceof Error ? e.stack : String(e)}\n`);
  process.exit(1);
});
