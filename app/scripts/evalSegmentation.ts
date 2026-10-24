/**
 * Accuracy of the ONLINE segmentation (online_v1, run by the real TrialController) against the
 * simulator's ground truth, for several base parameter sets. Nothing in src/ is changed: every set is
 * passed as the session's base params and each trial gets resolveParams(base, condition), as in the app
 * (the 'v1' set then lowers the slow onset floors back to protocol 1.0.0's 15 / 0.3).
 *
 * For N trials per (speed × variant × platform) cell (3 × 10 × 2 = 60 cells): a random simulated
 * participant (model.randomSimParticipant: 88% right-handed), a random hand, grip (all 10 orientation
 * ids, 'free' included) and start position, behaviour 'normal'. The SAME simulated trials (seeds) are
 * replayed for every parameter set, so differences are paired. Compared quantities, in the record's
 * time base (truth − timing.t0_perf_ms):
 *   onset error = timing.onset_ms − truth.onset_t (movement start; the online onset is back-dated)
 *   end error   = timing.end_ms − truth.end_t     (phone reaches the nose; the online end is back-dated)
 *   end cue lag = timing.end_detected_ms − truth.end_t (when the participant hears the end beep)
 * Gross errors: |onset error| > 100 ms, |end error| > 300 ms. Also: status counts, detection rate
 * (status complete with onset and end), GO time (pre-cue data), data kept after the true arrival
 * (stop − truth end), trial length, and the return of the arm (protocol 2.0.0 records it after the
 * end: return_onset_ms / return_end_ms); false starts ('false_start' behaviour: one anticipatory
 * twitch still_hold_ms + 200 ms after the start, i.e. inside each set's foreperiod), 'no_move'
 * trials, and a hold-still stress group (tremor × 2) per platform.
 *
 * Usage: npx tsx scripts/evalSegmentation.ts [--n 30] [--seed 7] [--out scripts/out/segmentation_eval.md]
 *          [--sets default,v1,...|all]   (default: DEFAULT_SETS below; 'all' adds the exploration sets)
 * Prints the summary tables and writes every table (also the per-cell ones) as Markdown.
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { ALL_ORIENTATIONS, DEFAULT_SEGMENTATION_PARAMS, ODD_VARIANTS } from '../src/core/constants';
import { resolveParams, SPEEDS, START_POSITIONS } from '../src/core/protocol';
import { createTrialController } from '../src/core/trialController';
import type { SegmentationParams, Speed, TrialCondition, TrialRecord, TrialStatus, Variant } from '../src/core/types';
import { makeRng, mixSeed } from '../src/core/util';
import { defaultSimPlatform, randomSimParticipant } from '../src/sim/model';
import { createSimulatedSource, type SimBehavior, type SimTruth } from '../src/sim/simulatedSource';

// ───────────────────────────── parameter sets ─────────────────────────────

interface ParamSet {
  name: string;
  note: string;
  params: SegmentationParams;
  /** per-trial params from the base (default: protocol.resolveParams, as the app does) */
  resolve?: (base: SegmentationParams, c: TrialCondition) => SegmentationParams;
}

const D = DEFAULT_SEGMENTATION_PARAMS;

/** Protocol 1.0.0's base parameters (constants.ts before 2.0.0): no return_cap_ms, so no return recording. */
const { return_cap_ms: _returnCap, ...D_NO_RETURN } = D;
const V1_BASE: SegmentationParams = {
  ...D_NO_RETURN,
  still_gyro_dps: 15,
  still_acc_ms2: 0.35,
  still_hold_ms: 600,
  foreperiod_min_ms: 500,
  foreperiod_max_ms: 1200,
  end_gyro_dps: 20,
  end_acc_ms2: 0.45,
  end_sustain_ms: 300,
  max_wait_ms: 6000,
  max_movement_ms: 8000,
  post_roll_ms: 1000,
};

/** Protocol 1.0.0's overrides = 2.0.0's with the slow onset floors lowered back to 15 deg/s / 0.3 m/s². */
function resolveV1(base: SegmentationParams, c: TrialCondition): SegmentationParams {
  const p = resolveParams(base, c);
  if (c.speed !== 'slow') return p;
  return { ...p, onset_gyro_min_dps: Math.min(p.onset_gyro_min_dps, 15), onset_acc_min_ms2: Math.min(p.onset_acc_min_ms2, 0.3) };
}

const ALL_SETS: ParamSet[] = [
  { name: 'default', note: 'DEFAULT_SEGMENTATION_PARAMS (constants.ts, protocol 2.0.0)', params: { ...D } },
  {
    name: 'v1',
    note:
      'protocol 1.0.0: still 15 / 0.35 / 600, foreperiod 500–1200, end 20 / 0.45 / 300, max_wait 6000, max_movement 8000, ' +
      'post_roll 1000, no return recording; slow onset floors 15 / 0.3',
    params: V1_BASE,
    resolve: resolveV1,
  },
  // ── exploration (select with --sets) ──
  {
    name: 'v2+v1slow',
    note: 'protocol 2.0.0 base with the protocol 1.0.0 slow onset floors (15 / 0.3)',
    params: { ...D },
    resolve: resolveV1,
  },
  { name: 'v2+wait6000', note: 'protocol 2.0.0 with max_wait 6000', params: { ...D, max_wait_ms: 6000 } },
  { name: 'v2+cap5000', note: 'protocol 2.0.0 with return_cap 5000', params: { ...D, return_cap_ms: 5000 } },
];

const DEFAULT_SETS = ['default', 'v1'];
let PARAM_SETS: ParamSet[] = ALL_SETS.filter((s) => DEFAULT_SETS.includes(s.name));

// ───────────────────────────── trial simulation ─────────────────────────────

const VARIANTS: Variant[] = ['standard', ...ODD_VARIANTS];
const PLATFORMS = ['ios', 'android'] as const;
type PlatformName = (typeof PLATFORMS)[number];

interface TrialSpec {
  key: string; // cell key
  speed: Speed;
  variant: Variant;
  platform: PlatformName;
  behavior: SimBehavior;
  seed: number;
  /** stress test of the hold-still criterion: physiological tremor × 2 */
  shaky?: boolean;
}

interface TrialEval {
  spec: TrialSpec;
  status: TrialStatus;
  onsetErr: number | null;
  endErr: number | null;
  endCueLag: number | null;
  goMs: number | null;
  stillMs: number | null; // hold-still criterion met (the one that scheduled the GO)
  stopMs: number;
  afterArrivalMs: number | null;
  falseStarts: number;
  twitched: boolean; // the simulated false start happened (before GO)
  /** the return of the arm (protocol 2.0.0): recorded onset / end, and the true return start (record time) */
  returnOnsetMs: number | null;
  returnEndMs: number | null;
  trueReturnMs: number | null;
  endDetectedMs: number | null;
}

const SILENT = { play() {} };

async function runTrial(spec: TrialSpec, set: ParamSet): Promise<TrialEval> {
  const base = set.params;
  const rng = makeRng(spec.seed);
  const drawn = randomSimParticipant(rng);
  const participant = spec.shaky ? { ...drawn, style: { ...drawn.style, tremor_dps: 2 * drawn.style.tremor_dps } } : drawn;
  const platform = defaultSimPlatform(spec.platform, rng);
  const condition: TrialCondition = {
    hand: rng() < 0.5 ? 'left' : 'right',
    orientation: ALL_ORIENTATIONS[Math.floor(rng() * ALL_ORIENTATIONS.length)],
    speed: spec.speed,
    start_position: START_POSITIONS[Math.floor(rng() * START_POSITIONS.length)],
    variant: spec.variant,
    posture: 'standing',
    rep: 0,
  };
  const simSeed = Math.floor(rng() * 4294967296) >>> 0;
  const fpSeed = Math.floor(rng() * 4294967296) >>> 0;
  const source = createSimulatedSource({
    mode: 'offline',
    participant,
    platform,
    condition,
    seed: simSeed,
    behavior: spec.behavior,
    startTime: 1000,
    baseEpochMs: 1_790_000_000_000,
    // the twitch comes inside this set's foreperiod (the hold is met still_hold_ms after the start)
    anticipationAfterMs: base.still_hold_ms + 200,
  });
  const ctrl = createTrialController({
    source,
    cues: SILENT,
    ids: {
      trial_id: 'eval',
      session_id: 'eval',
      participant_id: participant.participant_id,
      study_code: 'segmentation-eval',
      block_index: 0,
      trial_index: 0,
      trial_in_block: 0,
      attempt: 1,
    },
    condition,
    params: (set.resolve ?? resolveParams)(base, condition),
    calibration: null, // only used for detected.rotation_*, not for the segmentation
    rng: makeRng(fpSeed),
    appVersion: 'eval',
    simulated: true,
  });
  const t0 = source.clock.now();
  await ctrl.start();
  const rec: TrialRecord = await ctrl.result;
  const tr: SimTruth = source.truth;
  const rel = (v: number | null): number | null => (v === null ? null : v - t0);
  const onsetT = rel(tr.onset_t);
  const endT = rel(tr.end_t);
  const tm = rec.timing;
  return {
    spec,
    status: rec.outcome.status,
    onsetErr: tm.onset_ms !== null && onsetT !== null ? tm.onset_ms - onsetT : null,
    endErr: tm.end_ms !== null && endT !== null ? tm.end_ms - endT : null,
    endCueLag: tm.end_detected_ms !== null && endT !== null ? tm.end_detected_ms - endT : null,
    goMs: tm.go_ms,
    stillMs: tm.still_ms,
    stopMs: tm.stop_ms,
    afterArrivalMs: endT !== null && rec.outcome.status === 'complete' ? tm.stop_ms - endT : null,
    falseStarts: tm.false_starts,
    twitched: tr.false_start_t !== null && tr.false_start_t !== undefined,
    returnOnsetMs: tm.return_onset_ms ?? null,
    returnEndMs: tm.return_end_ms ?? null,
    trueReturnMs: rel(tr.return_t ?? null),
    endDetectedMs: tm.end_detected_ms,
  };
}

function cellSpecs(n: number, seed: number): TrialSpec[] {
  const out: TrialSpec[] = [];
  let cell = 0;
  for (const speed of SPEEDS) {
    for (const variant of VARIANTS) {
      for (const platform of PLATFORMS) {
        for (let k = 0; k < n; k++) {
          out.push({ key: `${speed}|${variant}|${platform}`, speed, variant, platform, behavior: 'normal', seed: mixSeed(mixSeed(seed, cell), k) });
        }
        cell++;
      }
    }
  }
  return out;
}

function behaviorSpecs(behavior: SimBehavior, n: number, seed: number, shaky = false): TrialSpec[] {
  const out: TrialSpec[] = [];
  const salt = (shaky ? 3000 : behavior === 'false_start' ? 1000 : 2000) + (behavior === 'no_move' ? 100 : 0);
  PLATFORMS.forEach((platform, j) => {
    for (let k = 0; k < n; k++) {
      out.push({
        key: `${behavior}${shaky ? '+shaky' : ''}|${platform}`,
        speed: 'natural',
        variant: 'standard',
        platform,
        behavior,
        seed: mixSeed(mixSeed(seed, salt + j), k),
        shaky,
      });
    }
  });
  return out;
}

// ───────────────────────────── statistics ─────────────────────────────

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

interface ErrStats {
  n: number;
  median: number;
  p25: number;
  p75: number;
  p95abs: number;
}

function errStats(values: (number | null)[]): ErrStats {
  const v = values.filter((x): x is number => x !== null).sort((a, b) => a - b);
  const abs = v.map(Math.abs).sort((a, b) => a - b);
  return { n: v.length, median: quantile(v, 0.5), p25: quantile(v, 0.25), p75: quantile(v, 0.75), p95abs: quantile(abs, 0.95) };
}

const fmt0 = (x: number): string => (Number.isFinite(x) ? (Math.round(x) === 0 ? '0' : x.toFixed(0)) : '–'); // no "-0"
const pct = (k: number, n: number): string => (n > 0 ? `${((100 * k) / n).toFixed(1)}%` : '–');
const errText = (s: ErrStats): string => (s.n ? `${fmt0(s.median)} [${fmt0(s.p25)}, ${fmt0(s.p75)}] · ${fmt0(s.p95abs)}` : '–');

interface Summary {
  n: number;
  complete: number;
  statuses: Record<TrialStatus, number>;
  detected: number; // complete with onset and end
  onset: ErrStats;
  end: ErrStats;
  cueLag: ErrStats;
  grossOnset: number;
  grossEnd: number;
  earlyEnd: number; // end declared > 300 ms before arrival
  go: number; // median GO time
  goP95: number;
  still: number; // median time at which the hold-still criterion was met
  stillP95: number;
  after: number; // median data after the true arrival
  stop: number; // median trial length
}

const GROSS_ONSET_MS = 100;
const GROSS_END_MS = 300;

function summarize(rows: TrialEval[]): Summary {
  const statuses: Record<TrialStatus, number> = { complete: 0, no_movement: 0, too_long: 0, aborted: 0, interrupted: 0 };
  for (const r of rows) statuses[r.status]++;
  const sorted = (xs: (number | null)[]) => xs.filter((x): x is number => x !== null).sort((a, b) => a - b);
  const med = (xs: (number | null)[]) => quantile(sorted(xs), 0.5);
  const q95 = (xs: (number | null)[]) => quantile(sorted(xs), 0.95);
  return {
    n: rows.length,
    complete: statuses.complete,
    statuses,
    detected: rows.filter((r) => r.status === 'complete' && r.onsetErr !== null && r.endErr !== null).length,
    onset: errStats(rows.map((r) => r.onsetErr)),
    end: errStats(rows.map((r) => r.endErr)),
    cueLag: errStats(rows.map((r) => r.endCueLag)),
    grossOnset: rows.filter((r) => r.onsetErr !== null && Math.abs(r.onsetErr) > GROSS_ONSET_MS).length,
    grossEnd: rows.filter((r) => r.endErr !== null && Math.abs(r.endErr) > GROSS_END_MS).length,
    earlyEnd: rows.filter((r) => r.endErr !== null && r.endErr < -GROSS_END_MS).length,
    go: med(rows.map((r) => r.goMs)),
    goP95: q95(rows.map((r) => r.goMs)),
    still: med(rows.map((r) => r.stillMs)),
    stillP95: q95(rows.map((r) => r.stillMs)),
    after: med(rows.map((r) => r.afterArrivalMs)),
    stop: med(rows.map((r) => r.stopMs)),
  };
}

function statusText(s: Summary): string {
  return (Object.keys(s.statuses) as TrialStatus[])
    .filter((k) => k !== 'complete' && s.statuses[k] > 0)
    .map((k) => `${k} ${s.statuses[k]}`)
    .join(', ');
}

// ───────────────────────────── tables ─────────────────────────────

function table(header: string[], rows: string[][]): string {
  // cells such as "p95 |e|" contain pipes, which would split Markdown table columns
  const line = (cells: string[]) => `| ${cells.map((c) => c.replace(/\|/g, '\\|')).join(' | ')} |`;
  return [line(header), line(header.map(() => '---')), ...rows.map(line)].join('\n');
}

const MAIN_HEADER = [
  'set',
  'n',
  'detected (complete, onset + end)',
  'other statuses',
  'onset err ms: median [IQR] · p95|e|',
  'end err ms: median [IQR] · p95|e|',
  `|onset err| > ${GROSS_ONSET_MS}`,
  `end err < −${GROSS_END_MS} (early) / > +${GROSS_END_MS} (late)`,
  'end cue after arrival (median)',
  'hold-still met at (median · p95)',
  'GO at (median · p95)',
  'kept after arrival (median)',
  'trial length (median)',
];

function mainRow(label: string, s: Summary): string[] {
  return [
    label,
    String(s.n),
    pct(s.detected, s.n),
    statusText(s) || '–',
    errText(s.onset),
    errText(s.end),
    pct(s.grossOnset, s.onset.n),
    `${pct(s.earlyEnd, s.end.n)} / ${pct(s.grossEnd - s.earlyEnd, s.end.n)}`,
    `${fmt0(s.cueLag.median)} ms`,
    `${fmt0(s.still)} · ${fmt0(s.stillP95)} ms`,
    `${fmt0(s.go)} · ${fmt0(s.goP95)} ms`,
    `${fmt0(s.after)} ms`,
    `${fmt0(s.stop)} ms`,
  ];
}

/** Median of per-trial differences set − reference (trials are paired: same spec, same index). */
function pairedMedian(a: TrialEval[], ref: TrialEval[], f: (r: TrialEval) => number | null): number {
  const d: number[] = [];
  a.forEach((r, i) => {
    const x = f(r);
    const y = f(ref[i]);
    if (x !== null && y !== null) d.push(x - y);
  });
  return quantile(d.sort((p, q) => p - q), 0.5);
}

function pairedRows(results: TrialEval[][]): string[][] {
  const ref = results[0];
  const s0 = summarize(ref);
  const signedFixed = (x: number, digits: number, unit: string): string => {
    if (!Number.isFinite(x)) return '–';
    const r = Number(x.toFixed(digits));
    return r === 0 ? `0${unit}` : `${r > 0 ? '+' : ''}${r.toFixed(digits)}${unit}`; // no "-0"
  };
  const signed = (x: number) => signedFixed(x, 0, ' ms');
  const pts = (x: number) => signedFixed(x, 1, ' pt');
  return PARAM_SETS.slice(1).map((p, j) => {
    const rows = results[j + 1];
    const s = summarize(rows);
    return [
      p.name,
      signed(pairedMedian(rows, ref, (r) => r.goMs)),
      signed(pairedMedian(rows, ref, (r) => r.afterArrivalMs)),
      signed(pairedMedian(rows, ref, (r) => r.endCueLag)),
      signed(pairedMedian(rows, ref, (r) => r.stopMs)),
      signed(pairedMedian(rows, ref, (r) => r.onsetErr)),
      signed(s.onset.p95abs - s0.onset.p95abs),
      signed(pairedMedian(rows, ref, (r) => r.endErr)),
      pts((100 * s.grossOnset) / s.onset.n - (100 * s0.grossOnset) / s0.onset.n),
      pts((100 * s.earlyEnd) / s.end.n - (100 * s0.earlyEnd) / s0.end.n),
      pts((100 * s.detected) / s.n - (100 * s0.detected) / s0.n),
    ];
  });
}

const PAIRED_HEADER = [
  'set (− reference, paired median)',
  'Δ GO time',
  'Δ kept after arrival',
  'Δ end cue lag',
  'Δ trial length',
  'Δ onset err',
  'Δ p95 |onset err|',
  'Δ end err',
  `Δ |onset err| > ${GROSS_ONSET_MS}`,
  `Δ early end (< −${GROSS_END_MS})`,
  'Δ detected',
];

function groupHeader(): string[] {
  return ['group', ...PARAM_SETS.map((p) => `${p.name}: complete · onset median/p95 · end median/p95 · |end| > ${GROSS_END_MS}`)];
}

function groupRows(results: TrialEval[][], groups: { label: string; filter: (r: TrialEval) => boolean }[]): string[][] {
  return groups.map((g) => [
    g.label,
    ...results.map((rows) => {
      const s = summarize(rows.filter(g.filter));
      return `${pct(s.complete, s.n)} · ${fmt0(s.onset.median)}/${fmt0(s.onset.p95abs)} · ${fmt0(s.end.median)}/${fmt0(s.end.p95abs)} · ${pct(s.grossEnd, s.end.n)}`;
    }),
  ]);
}

// ───────────────────────────── main ─────────────────────────────

function argValue(name: string): string | undefined {
  const argv = process.argv.slice(2);
  for (let k = 0; k < argv.length; k++) {
    if (argv[k] === `--${name}`) return argv[k + 1];
    if (argv[k].startsWith(`--${name}=`)) return argv[k].slice(name.length + 3);
  }
  return undefined;
}

async function main(): Promise<void> {
  const n = Number(argValue('n') ?? 30);
  const seed = Number(argValue('seed') ?? 7) >>> 0;
  const out = resolve(process.cwd(), argValue('out') ?? 'scripts/out/segmentation_eval.md');
  if (!Number.isInteger(n) || n < 1) throw new Error('--n must be a positive integer');
  const wanted = argValue('sets');
  if (wanted) {
    const names = wanted === 'all' ? ALL_SETS.map((s) => s.name) : wanted.split(',');
    const unknown = names.filter((x) => !ALL_SETS.some((s) => s.name === x));
    if (unknown.length) throw new Error(`unknown --sets ${unknown.join(', ')}; known: ${ALL_SETS.map((s) => s.name).join(', ')}`);
    PARAM_SETS = ALL_SETS.filter((s) => names.includes(s.name));
  }
  const specs = cellSpecs(n, seed);
  const fsSpecs = behaviorSpecs('false_start', 4 * n, seed);
  const nmSpecs = behaviorSpecs('no_move', n, seed);
  const shSpecs = behaviorSpecs('normal', 2 * n, seed, true);
  const started = Date.now();

  const run = async (list: TrialSpec[], set: ParamSet): Promise<TrialEval[]> => {
    const rows: TrialEval[] = [];
    for (const s of list) rows.push(await runTrial(s, set));
    return rows;
  };
  const results: TrialEval[][] = [];
  const fsResults: TrialEval[][] = [];
  const nmResults: TrialEval[][] = [];
  const shResults: TrialEval[][] = [];
  for (const ps of PARAM_SETS) {
    const t1 = Date.now();
    results.push(await run(specs, ps));
    fsResults.push(await run(fsSpecs, ps));
    nmResults.push(await run(nmSpecs, ps));
    shResults.push(await run(shSpecs, ps));
    const total = specs.length + fsSpecs.length + nmSpecs.length + shSpecs.length;
    console.log(`${ps.name}: ${total} trials in ${((Date.now() - t1) / 1000).toFixed(1)} s`);
  }

  const md: string[] = [];
  md.push('# Online segmentation vs simulator truth');
  md.push('');
  md.push(
    `Generated by \`app/scripts/evalSegmentation.ts\` (n = ${n} per speed × variant × platform cell, seed ${seed}; ` +
      `${specs.length} normal trials per parameter set, plus ${fsSpecs.length} false-start, ${nmSpecs.length} no-move and ` +
      `${shSpecs.length} tremor × 2 trials). ` +
      'The same simulated trials are replayed for every parameter set. Errors are online − truth in ms (negative = the detector is early). ' +
      'Onset truth = movement start, end truth = phone reaches the nose. Every trial runs with resolveParams(base, condition), so the ' +
      'slow/fast/pause/jerky overrides of protocol.ts apply on top of each base set (the v1 set lowers the slow onset floors back to ' +
      '15 / 0.3). Android-like trials are Chromium-quantised. The simulated participant brings the arm back out 250–450 ms after ' +
      'the end cue (0.8–1.2 s); "trial length" includes that return wherever the set records it (return_cap_ms).',
  );
  md.push('');
  md.push('## Parameter sets');
  md.push('');
  md.push(table(['set', 'base parameters'], PARAM_SETS.map((p) => [p.name, p.note])));
  md.push('');
  md.push('## Overall (normal behaviour)');
  md.push('');
  md.push(table(MAIN_HEADER, PARAM_SETS.map((p, k) => mainRow(p.name, summarize(results[k])))));
  md.push('');
  if (PARAM_SETS.length > 1) {
    md.push(`## Paired differences against \`${PARAM_SETS[0].name}\``);
    md.push('');
    md.push('Median over trials of (set − reference) for the same simulated trial; rates in percentage points.');
    md.push('');
    md.push(table(PAIRED_HEADER, pairedRows(results)));
    md.push('');
  }
  md.push('## By platform');
  md.push('');
  md.push(
    table(
      MAIN_HEADER,
      PARAM_SETS.flatMap((p, k) => PLATFORMS.map((pl) => mainRow(`${p.name} · ${pl}`, summarize(results[k].filter((r) => r.spec.platform === pl))))),
    ),
  );
  md.push('');
  md.push('## By speed (all variants and platforms)');
  md.push('');
  md.push(table(groupHeader(), groupRows(results, SPEEDS.map((s) => ({ label: s, filter: (r: TrialEval) => r.spec.speed === s })))));
  md.push('');
  md.push('## By variant (all speeds and platforms)');
  md.push('');
  md.push(table(groupHeader(), groupRows(results, VARIANTS.map((v) => ({ label: v, filter: (r: TrialEval) => r.spec.variant === v })))));
  md.push('');
  md.push('## Return of the arm (complete normal trials)');
  md.push('');
  md.push(
    'Protocol 2.0.0 keeps recording after the end until the arm is back out and still (or return_cap_ms after the end was ' +
      'confirmed): timing.return_onset_ms / return_end_ms. "onset error" = recorded return onset − true start of the return.',
  );
  md.push('');
  md.push(
    table(
      ['set', 'complete', 'return onset recorded', 'return end recorded', 'return onset err ms (median · p95|e|)', 'end cue → stop (median)'],
      PARAM_SETS.map((p, k) => {
        const done = results[k].filter((r) => r.status === 'complete');
        const withOnset = done.filter((r) => r.returnOnsetMs !== null);
        const err = errStats(withOnset.map((r) => (r.trueReturnMs === null ? null : (r.returnOnsetMs as number) - r.trueReturnMs)));
        const tail = done.map((r) => (r.endDetectedMs === null ? null : r.stopMs - r.endDetectedMs)).filter((x): x is number => x !== null);
        return [
          p.name,
          String(done.length),
          pct(withOnset.length, done.length),
          pct(done.filter((r) => r.returnEndMs !== null).length, done.length),
          err.n ? `${fmt0(err.median)} · ${fmt0(err.p95abs)}` : '–',
          `${fmt0(quantile(tail.sort((a, b) => a - b), 0.5))} ms`,
        ];
      }),
    ),
  );
  md.push('');
  md.push('## False starts and no-move trials');
  md.push('');
  md.push(
    'false_start: the simulated participant twitches still_hold_ms + 200 ms after the trial starts (inside the foreperiod of every ' +
      'set, before its earliest possible GO). "counted" = share of those twitches with timing.false_starts ≥ 1 (the controller ' +
      'cancelled the pending GO, played the error cue and restarted the hold-still); the rest were taken as silent losses of ' +
      'stillness or not noticed. The simulated twitch is deliberately large (≈ 3.8 m/s² step), so this checks the handling, not ' +
      'the sensitivity, of the false-start detector.',
  );
  md.push('');
  md.push(
    table(
      ['set · platform', 'n', 'twitch before GO', 'counted as false start', 'complete', 'onset err median · p95|e|', 'GO at (median)', 'no_move → no_movement'],
      PARAM_SETS.flatMap((p, k) =>
        PLATFORMS.map((pl) => {
          const fs = fsResults[k].filter((r) => r.spec.platform === pl);
          const nm = nmResults[k].filter((r) => r.spec.platform === pl);
          const tw = fs.filter((r) => r.twitched);
          const s = summarize(fs);
          return [
            `${p.name} · ${pl}`,
            String(fs.length),
            pct(tw.length, fs.length),
            pct(tw.filter((r) => r.falseStarts >= 1).length, tw.length),
            pct(s.complete, s.n),
            `${fmt0(s.onset.median)} · ${fmt0(s.onset.p95abs)}`,
            `${fmt0(s.go)} ms`,
            pct(nm.filter((r) => r.status === 'no_movement').length, nm.length),
          ];
        }),
      ),
    ),
  );
  md.push('');
  md.push('## Hold-still robustness: participants with twice the simulated tremor');
  md.push('');
  md.push(
    'Natural speed, standard variant, random grips. Shows how much margin each set\'s still thresholds leave: the time at which the ' +
      'hold-still criterion was met (a late value means the run was interrupted by tremor), the GO time and the outcome.',
  );
  md.push('');
  md.push(
    table(
      MAIN_HEADER,
      PARAM_SETS.flatMap((p, k) => PLATFORMS.map((pl) => mainRow(`${p.name} · ${pl}`, summarize(shResults[k].filter((r) => r.spec.platform === pl))))),
    ),
  );
  md.push('');
  md.push('## Every cell');
  md.push('');
  md.push('Per set: complete · onset median/p95|e| · end median/p95|e| (ms).');
  md.push('');
  const cellGroups: { label: string; filter: (r: TrialEval) => boolean }[] = [];
  for (const speed of SPEEDS) {
    for (const variant of VARIANTS) {
      for (const platform of PLATFORMS) {
        const key = `${speed}|${variant}|${platform}`;
        cellGroups.push({ label: `${speed} · ${variant} · ${platform}`, filter: (r) => r.spec.key === key });
      }
    }
  }
  md.push(table(groupHeader(), groupRows(results, cellGroups)));
  md.push('');

  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, md.join('\n'));
  const overall = PARAM_SETS.map((p, k) => mainRow(p.name, summarize(results[k])));
  console.log(table(MAIN_HEADER, overall));
  console.log('');
  if (PARAM_SETS.length > 1) {
    console.log(`paired differences against ${PARAM_SETS[0].name}:`);
    console.log(table(PAIRED_HEADER, pairedRows(results)));
    console.log('');
  }
  console.log('tremor × 2 (hold-still robustness):');
  console.log(table(MAIN_HEADER, PARAM_SETS.map((p, k) => mainRow(p.name, summarize(shResults[k])))));
  console.log('');
  console.log(table(groupHeader(), groupRows(results, SPEEDS.map((s) => ({ label: s, filter: (r: TrialEval) => r.spec.speed === s })))));
  console.log('');
  console.log(table(groupHeader(), groupRows(results, VARIANTS.map((v) => ({ label: v, filter: (r: TrialEval) => r.spec.variant === v })))));
  console.log(`\nwritten: ${out} (${((Date.now() - started) / 1000).toFixed(1)} s)`);
}

main().catch((e: unknown) => {
  console.error('evalSegmentation failed:', e);
  process.exit(1);
});
