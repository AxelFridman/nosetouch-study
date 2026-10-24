// Regenerates street_plans_ts.json.gz: 'street' plans (protocol 2.0.0) built by the app's OWN
// protocol.ts, the ground truth of the Python port motionlab.synth.build_street_plan
// (analysis/tests/test_street_plan.py). Run it whenever app/src/core/protocol.ts or util.ts change:
//   cd app && npx tsx ../analysis/tests/fixtures/make_street_plans.mts
// Contents: { generated_by, protocol_version, plans: [{ seed, first_hand, plan: PlannedBlock[], minutes }] }.
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, '../../../app');
const OUT = path.join(HERE, 'street_plans_ts.json.gz');
const load = (p: string) => import(pathToFileURL(path.join(APP, p)).href);

const { buildPlan, estimateMinutes } = await load('src/core/protocol.ts');
const { PROTOCOL_VERSION } = await load('src/core/constants.ts');

// seeds 0-3 cover the four (first hand, first sideways side) combinations; the large ones exercise the
// uint32 arithmetic of mulberry32 / mixSeed (2^31 + 5, 2^32 - 1); one firstHand override and one sitting plan
const cases: Array<{ seed: number; first_hand?: 'left' | 'right'; posture?: 'standing' | 'sitting' }> = [
  { seed: 0 },
  { seed: 1 },
  { seed: 2 },
  { seed: 3 },
  { seed: 12345 },
  { seed: 2147483653 },
  { seed: 4294967295 },
  { seed: 42, first_hand: 'left', posture: 'sitting' },
];
const plans = cases.map(({ seed, first_hand, posture }) => {
  const plan = buildPlan({ preset: 'street', seed, posture: posture ?? 'standing', ...(first_hand ? { firstHand: first_hand } : {}) });
  return { seed, first_hand: first_hand ?? null, posture: posture ?? 'standing', plan, minutes: estimateMinutes(plan) };
});
writeFileSync(OUT, gzipSync(JSON.stringify({ generated_by: 'analysis/tests/fixtures/make_street_plans.mts', protocol_version: PROTOCOL_VERSION, plans })));
console.log(`wrote ${OUT}: ${plans.length} plans`);
