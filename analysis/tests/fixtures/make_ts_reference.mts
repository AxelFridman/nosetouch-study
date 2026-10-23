// Regenerates ts_online_v1_reference.json.gz: ground truth from the app's OWN TypeScript code for the
// Python online_v1 port (analysis/tests/test_segmentation.py). Run it whenever app/src/core/segmentation.ts,
// trialController.ts or the simulator change:
//   cd app && npx tsx ../analysis/tests/fixtures/make_ts_reference.mts
// Contents: { records: TrialRecord[] (real simulator -> TrialController, both platforms, every outcome),
//             scenarios: [{ motion, params, go_t, foreperiod_ms, ts: runOnlineSegmentation(...) }] }.
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, '../../../app');
const OUT = path.join(HERE, 'ts_online_v1_reference.json.gz');
const load = (p: string) => import(pathToFileURL(path.join(APP, p)).href);

const { createSimulatedSource } = await load('src/sim/simulatedSource.ts');
const { randomSimParticipant, defaultSimPlatform } = await load('src/sim/model.ts');
const { makeRng } = await load('src/core/util.ts');
const { resolveParams } = await load('src/core/protocol.ts');
const { DEFAULT_SEGMENTATION_PARAMS } = await load('src/core/constants.ts');
const { createTrialController } = await load('src/core/trialController.ts');
const { runOnlineSegmentation } = await load('src/core/segmentation.ts');

type Plan = { p: number; platform: 'ios' | 'android'; behavior: string; speed: string; variant: string; orientation: string; accOff?: boolean };
const plans: Plan[] = [
  { p: 0, platform: 'android', behavior: 'normal', speed: 'natural', variant: 'standard', orientation: 'upright_toward' },
  { p: 0, platform: 'android', behavior: 'normal', speed: 'fast', variant: 'standard', orientation: 'top_left_toward' },
  { p: 0, platform: 'android', behavior: 'normal', speed: 'slow', variant: 'pause', orientation: 'upright_away' },
  { p: 1, platform: 'ios', behavior: 'normal', speed: 'natural', variant: 'jerky', orientation: 'top_right_toward' },
  { p: 1, platform: 'ios', behavior: 'false_start', speed: 'natural', variant: 'standard', orientation: 'upright_toward' },
  { p: 1, platform: 'ios', behavior: 'no_move', speed: 'natural', variant: 'standard', orientation: 'flat_up' },
  { p: 2, platform: 'android', behavior: 'too_slow', speed: 'natural', variant: 'standard', orientation: 'upside_down_toward' },
  { p: 2, platform: 'android', behavior: 'normal', speed: 'slow', variant: 'standard', orientation: 'free', accOff: true },
  { p: 2, platform: 'android', behavior: 'no_move', speed: 'fast', variant: 'standard', orientation: 'upright_toward', accOff: true },
  { p: 3, platform: 'ios', behavior: 'normal', speed: 'fast', variant: 'twist', orientation: 'top_left_away' },
  { p: 3, platform: 'ios', behavior: 'too_slow', speed: 'slow', variant: 'standard', orientation: 'upright_toward' },
  { p: 3, platform: 'ios', behavior: 'normal', speed: 'natural', variant: 'curved', orientation: 'upside_down_away' },
];

const records: any[] = [];
for (let k = 0; k < plans.length; k++) {
  const pl = plans[k];
  const participant = randomSimParticipant(makeRng(1000 + pl.p));
  const platform = { ...defaultSimPlatform(pl.platform, makeRng(2000 + pl.p)), ...(pl.accOff ? { acc_available: false } : {}) };
  const condition = {
    hand: k % 2 ? 'left' : 'right', orientation: pl.orientation, speed: pl.speed, start_position: 'natural',
    variant: pl.variant, posture: 'standing', rep: 0,
  };
  const params = resolveParams(DEFAULT_SEGMENTATION_PARAMS, condition);
  const source = createSimulatedSource({ mode: 'offline', participant, platform, condition, seed: 70_000 + k, behavior: pl.behavior });
  const ctrl = createTrialController({
    source,
    cues: { play: () => {} },
    ids: {
      trial_id: `ts-ref-${k}`, session_id: `ts-ref-session-${pl.p}`, participant_id: 'P-ABCDE2', study_code: 'ts-reference',
      block_index: 0, trial_index: k, trial_in_block: k, attempt: 1,
    },
    condition,
    params,
    calibration: {
      performed_at: '2026-10-23T10:00:00.000Z', flat: null, upright: null, gravity_sign: platform.gravitySign,
      gravity_magnitude: 9.81, upright_check: 'ok', gyro_bias_dps: null, notes: [],
    },
    rng: makeRng(9000 + k),
    appVersion: '0.1.0',
    simulated: true,
  });
  await ctrl.start();
  records.push(await ctrl.result);
}

// Stored streams re-segmented with variants the controller never produces: tau = 0, GO between two
// samples, automatic GO (goT null) with a foreperiod.
const samplesOf = (rec: any) => {
  const m = rec.streams.motion;
  const vec = (a: any, b: any, c: any) => (a !== null && b !== null && c !== null ? [a, b, c] : null);
  return m.t.map((t: number, i: number) => ({
    t, ts: m.ts[i], index: i, interval: null,
    acc: vec(m.acc_x[i], m.acc_y[i], m.acc_z[i]),
    accg: vec(m.accg_x[i], m.accg_y[i], m.accg_z[i]),
    rr: m.rr_alpha[i] !== null && m.rr_beta[i] !== null && m.rr_gamma[i] !== null ? { alpha: m.rr_alpha[i], beta: m.rr_beta[i], gamma: m.rr_gamma[i] } : null,
  }));
};
const scenarios: any[] = [];
for (const k of [0, 3, 7]) {
  const rec = records[k];
  const p = rec.segmentation.params;
  const go = rec.timing.go_ms;
  const variants: Array<[any, number | null, number | undefined]> = [
    [{ ...p, smoothing_ms: 0, gravity_tau_ms: 0 }, go, undefined],
    [p, go === null ? null : go + 3.7, undefined],
    [p, null, 500],
  ];
  for (const [params, goT, fp] of variants) {
    const ts = runOnlineSegmentation(samplesOf(rec), params, goT, fp);
    scenarios.push({ id: `${rec.trial_id}|tau0=${params.smoothing_ms === 0}|go=${goT}|fp=${fp ?? null}`, motion: rec.streams.motion, params, go_t: goT, foreperiod_ms: fp ?? null, ts });
  }
}

writeFileSync(OUT, gzipSync(JSON.stringify({ generated_by: 'analysis/tests/fixtures/make_ts_reference.mts', records, scenarios })));
console.log(`wrote ${records.length} records, ${scenarios.length} scenarios to ${OUT}`);
