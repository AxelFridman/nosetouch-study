/**
 * Generates a synthetic dataset with the REAL recording pipeline (protocol → simulated participant →
 * TrialController → TrialRecord; see src/sim/dataset.ts) and writes it in the backend's folder layout:
 *   <out>/<study_code>/<participant_id>/<session_id>/session.json
 *   <out>/<study_code>/<participant_id>/<session_id>/trial_<idx:3>_a<attempt>_<trial_id>.json
 * Usage: npm run synth -- --participants 12 --preset street --seed 42 --left-fraction 0.3 \
 *          --out ../analysis/data/synthetic [--study synthetic-v1] [--failure-rate 0.03] [--validate]
 *          [--ios-gravity-sign -1] [--clean] [--review-mode auto] [--calibration implicit] [--profile short]
 *          [--sway-rate 0.05] [--late-return-rate 0.05] [--bonus-fraction 0.5] [--end-early-fraction 0]
 *   --participants N         simulated participants, one session each (default 12)
 *   --preset P               street (default: the v2 visit, 29 trials + a 10-trial bonus) | quick |
 *                            standard | full | custom (custom = protocol.DEFAULT_CUSTOM_SPEC)
 *   --seed N                 uint32 master seed (default 42): same seed → byte-identical files
 *   --left-fraction F        share of left-handed participants (default 0.3, rounded to whole people)
 *   --failure-rate F         share of attempts performed as 'no_move' / 'false_start' (default 0.03)
 *   --sway-rate F            share of attempts with a posture sway before GO (silent; default 0.05)
 *   --late-return-rate F     share of attempts after which the participant lingers at the nose past
 *                            return_cap_ms, so the next chained trial awaits the return (default 0.05)
 *   --review-mode M          auto (default: hands-free, auto-accept) | tap (review_each_trial)
 *   --calibration C          implicit (default: implicit_v1) | guided (the v1 calibration screen)
 *   --profile P              short (default: the v2 "About you" form, names uploaded) | full (v1, EHI)
 *   --bonus-fraction F       share of participants taking the bonus at the checkpoint (default 0.5)
 *   --end-early-fraction F   share of participants who tap "Finish now" early → ended_early (default 0)
 *   --study CODE             study_code (default synthetic-v1); never a real study's code
 *   --out DIR                output root (default ../analysis/data/synthetic, relative to the cwd)
 *   --validate               validate every record with ajv against schema/records.schema.json
 *   --ios-gravity-sign S     acc/accg sign of the iOS-like devices: -1 (WebKit, default) or 1
 *   --clean                  delete <out>/<study_code>/ first (otherwise other runs' files stay)
 * Every record has simulated: true. Platforms alternate iOS-like (WebKit: gravity sign −1, full
 * precision, no quaternion) and Android-like (Chrome: +1, values quantised to 0.1 units, relative and
 * absolute quaternion, deviceorientationabsolute); the short profile holds a synthetic name, the
 * full one EHI answers consistent with the simulated handedness.
 * Prints a one-line summary per participant and the totals: sessions by status, attempt outcomes,
 * median attempt length (stop_ms) by speed for complete attempts, and how often the return of the
 * arm was recorded. Exit code 1 if a record fails validation.
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ajv2020 } from 'ajv/dist/2020.js';
import type { ProtocolPreset, ReviewMode, SessionStatus, Speed, TrialStatus } from '../src/core/types';
import { iterateDataset, resolveDatasetOptions, summarizeSession, type DatasetOptions } from '../src/sim/dataset';

const FLAGS = new Set(['validate', 'clean', 'help']);
const VALUED = new Set([
  'participants',
  'preset',
  'seed',
  'left-fraction',
  'failure-rate',
  'sway-rate',
  'late-return-rate',
  'review-mode',
  'calibration',
  'profile',
  'bonus-fraction',
  'end-early-fraction',
  'study',
  'out',
  'ios-gravity-sign',
]);

function usage(): string {
  const lines = readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n');
  const end = lines.findIndex((l) => l.startsWith(' */'));
  return lines
    .slice(1, end)
    .map((l) => l.replace(/^ \* ?/, ''))
    .join('\n');
}

/** `--name value` and `--name=value` (so negative numbers work: `--ios-gravity-sign -1`). */
function parseCli(argv: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let k = 0; k < argv.length; k++) {
    const arg = argv[k];
    if (!arg.startsWith('--')) throw new Error(`unexpected argument "${arg}"`);
    const eq = arg.indexOf('=');
    const name = arg.slice(2, eq < 0 ? undefined : eq);
    if (FLAGS.has(name)) {
      if (eq >= 0) throw new Error(`--${name} takes no value`);
      out.set(name, 'true');
    } else if (VALUED.has(name)) {
      const value = eq >= 0 ? arg.slice(eq + 1) : argv[++k];
      if (value === undefined) throw new Error(`--${name} needs a value`);
      out.set(name, value);
    } else {
      throw new Error(`unknown option --${name}`);
    }
  }
  return out;
}

function num(args: Map<string, string>, name: string): number | undefined {
  const v = args.get(name);
  if (v === undefined) return undefined;
  const n = Number(v);
  if (v.trim() === '' || !Number.isFinite(n)) throw new Error(`--${name} must be a number, got "${v}"`);
  return n;
}

function mb(bytes: number): string {
  return `${(bytes / 1e6).toFixed(2)} MB`;
}

/** numpy-style median (linear interpolation); NaN for an empty list. */
function median(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  const v = [...values].sort((a, b) => a - b);
  const pos = (v.length - 1) / 2;
  return v[Math.floor(pos)] + (v[Math.ceil(pos)] - v[Math.floor(pos)]) * (pos - Math.floor(pos));
}

function pct(k: number, n: number): string {
  return n > 0 ? `${((100 * k) / n).toFixed(1)} %` : '–';
}

const STATUSES: TrialStatus[] = ['complete', 'no_movement', 'too_long', 'aborted', 'interrupted'];
const SESSION_STATUSES: SessionStatus[] = ['completed', 'ended_early', 'abandoned', 'in_progress'];
const SPEEDS: Speed[] = ['natural', 'fast', 'slow'];

async function main(): Promise<number> {
  let args: Map<string, string>;
  let opts: DatasetOptions;
  try {
    args = parseCli(process.argv.slice(2));
    if (args.has('help')) {
      console.log(usage());
      return 0;
    }
    const sign = num(args, 'ios-gravity-sign');
    if (sign !== undefined && sign !== 1 && sign !== -1) throw new Error('--ios-gravity-sign must be 1 or -1');
    opts = {
      participants: num(args, 'participants'),
      preset: args.get('preset') as ProtocolPreset | undefined,
      seed: num(args, 'seed'),
      leftFraction: num(args, 'left-fraction'),
      failureRate: num(args, 'failure-rate'),
      swayRate: num(args, 'sway-rate'),
      lateReturnRate: num(args, 'late-return-rate'),
      reviewMode: args.get('review-mode') as ReviewMode | undefined,
      calibration: args.get('calibration') as DatasetOptions['calibration'],
      profile: args.get('profile') as DatasetOptions['profile'],
      bonusFraction: num(args, 'bonus-fraction'),
      endEarlyFraction: num(args, 'end-early-fraction'),
      studyCode: args.get('study'),
      iosGravitySign: sign as 1 | -1 | undefined,
    };
    const seed = opts.seed;
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)) {
      throw new Error('--seed must be an integer in [0, 4294967295]');
    }
    resolveDatasetOptions(opts); // reject bad values before anything is written
  } catch (e) {
    console.error(`synth: ${e instanceof Error ? e.message : String(e)}\n\n${usage()}`);
    return 2;
  }
  const o = resolveDatasetOptions(opts);
  const root = resolve(process.cwd(), args.get('out') ?? '../analysis/data/synthetic');
  const studyDir = join(root, o.studyCode);
  if (args.has('clean')) rmSync(studyDir, { recursive: true, force: true });
  else if (existsSync(studyDir) && readdirSync(studyDir).length > 0) {
    console.warn(`note: ${studyDir} already holds data; files from other runs are kept (use --clean to start empty)`);
  }

  let validate: ((kind: 'SessionRecord' | 'TrialRecord', record: unknown) => string | null) | null = null;
  if (args.has('validate')) {
    const schemaPath = fileURLToPath(new URL('../../schema/records.schema.json', import.meta.url));
    const schema = JSON.parse(readFileSync(schemaPath, 'utf8')) as { $id: string };
    const ajv = new Ajv2020({ allErrors: true, strict: false });
    ajv.addSchema(schema);
    validate = (kind, record) => {
      const fn = ajv.getSchema(`${schema.$id}#/$defs/${kind}`);
      if (!fn) throw new Error(`schema def ${kind} not found`);
      if (fn(record)) return null;
      return (fn.errors ?? [])
        .slice(0, 5)
        .map((e) => `${e.instancePath || '(root)'} ${e.message ?? ''}`)
        .join('; ');
    };
  }

  console.log(
    `synth: ${o.participants} participants, preset ${o.preset}, seed ${o.seed}, left fraction ${o.leftFraction}, ` +
      `failure rate ${o.failureRate}, sway rate ${o.swayRate}, late-return rate ${o.lateReturnRate}, review ${o.reviewMode}, ` +
      `calibration ${o.calibration}, profile ${o.profile}, bonus fraction ${o.bonusFraction}, end-early fraction ` +
      `${o.endEarlyFraction}, study ${o.studyCode}, iOS gravity sign ${o.iosGravitySign} → ${studyDir}`,
  );
  const started = Date.now();
  const totals = { files: 0, bytes: 0, trials: 0, accepted: 0, falseStarts: 0, invalid: 0, bonus: 0, rejects: 0 };
  const statusTotals = Object.fromEntries(STATUSES.map((s) => [s, 0])) as Record<TrialStatus, number>;
  const sessionTotals = Object.fromEntries(SESSION_STATUSES.map((s) => [s, 0])) as Record<SessionStatus, number>;
  const behaviorTotals: Record<string, number> = {};
  const byPlatform = { ios: 0, android: 0 };
  /** stop_ms of complete attempts, by speed */
  const lengths: Record<Speed, number[]> = { natural: [], fast: [], slow: [] };
  /** complete attempts, those with the return of the arm recorded, and those whose participant lingered on purpose */
  const ret = { complete: 0, onset: 0, end: 0, lingered: 0 };
  const minutes: number[] = [];
  let nLeft = 0;
  let t1 = Date.now(); // per-participant wall time: generation + writing

  for await (const s of iterateDataset(o)) {
    const dir = join(studyDir, s.session.participant_id, s.session.session_id);
    mkdirSync(dir, { recursive: true });
    let bytes = 0;
    const write = (name: string, kind: 'SessionRecord' | 'TrialRecord', record: unknown) => {
      const problem = validate?.(kind, record) ?? null;
      if (problem) {
        totals.invalid++;
        if (totals.invalid <= 5) console.error(`INVALID ${kind} ${name}: ${problem}`);
      }
      const text = JSON.stringify(record);
      writeFileSync(join(dir, name), text);
      bytes += Buffer.byteLength(text);
      totals.files++;
    };
    write('session.json', 'SessionRecord', s.session);
    for (const t of s.trials) {
      write(`trial_${String(t.trial_index).padStart(3, '0')}_a${t.attempt}_${t.trial_id}.json`, 'TrialRecord', t);
    }
    totals.bytes += bytes;

    const sum = summarizeSession(s);
    totals.trials += s.trials.length;
    totals.accepted += sum.n_accepted;
    totals.falseStarts += sum.false_starts;
    totals.rejects += sum.n_orientation_rejects;
    if (sum.bonus) totals.bonus++;
    sessionTotals[sum.status]++;
    minutes.push(sum.minutes);
    for (const st of STATUSES) statusTotals[st] += sum.statuses[st];
    s.trials.forEach((t, k) => {
      if (t.outcome.status !== 'complete') return;
      lengths[t.condition.speed].push(t.timing.stop_ms);
      ret.complete++;
      if (s.attempts[k].late_return) ret.lingered++;
      if (t.timing.return_onset_ms != null) ret.onset++;
      if (t.timing.return_end_ms != null) ret.end++;
    });
    for (const [b, n] of Object.entries(sum.behaviors)) behaviorTotals[b] = (behaviorTotals[b] ?? 0) + (n ?? 0);
    byPlatform[sum.platform]++;
    if (sum.dominant === 'left') nLeft++;
    const failed = STATUSES.filter((st) => st !== 'complete' && sum.statuses[st] > 0)
      .map((st) => `${st} ${sum.statuses[st]}`)
      .join(', ');
    const ehi = sum.ehi_class === null ? '' : ` (EHI ${sum.ehi_class})`;
    const bonus = sum.bonus === null ? '' : sum.bonus ? ', bonus' : ', no bonus';
    console.log(
      `${String(s.design.index + 1).padStart(3)}/${o.participants} ${sum.participant_id} ${sum.platform.padEnd(7)} ` +
        `${`${sum.dominant}-handed${ehi}`.padEnd(sum.ehi_class === null ? 12 : 23)} ${sum.sex} ${sum.age_years ?? '?'} y ` +
        `${sum.height_cm ?? '?'} cm, first ${sum.first_hand}, gravity sign ${sum.gravity_sign ?? '?'}: ` +
        `${sum.n_accepted}/${sum.n_planned} accepted in ${sum.n_attempts} attempts${failed ? ` (${failed})` : ''}, ` +
        `false starts ${sum.false_starts}, return not seen ${sum.n_return_missed}${bonus}, ${sum.status}, ` +
        `${sum.minutes} min, ${mb(bytes)}, ${((Date.now() - t1) / 1000).toFixed(1)} s`,
    );
    t1 = Date.now();
  }

  const seconds = (Date.now() - started) / 1000;
  const statusText = STATUSES.filter((st) => statusTotals[st] > 0)
    .map((st) => `${st} ${statusTotals[st]}`)
    .join(', ');
  const behaviorText = Object.entries(behaviorTotals)
    .map(([b, n]) => `${b} ${n}`)
    .join(', ');
  const sessionText = SESSION_STATUSES.filter((st) => sessionTotals[st] > 0)
    .map((st) => `${sessionTotals[st]} ${st}`)
    .join(', ');
  console.log(
    `total: ${o.participants} participants (${byPlatform.ios} iOS-like, ${byPlatform.android} Android-like; ${nLeft} left-handed), ` +
      `sessions ${sessionText}${totals.bonus ? ` (${totals.bonus} with the bonus)` : ''}; ${totals.trials} trial records ` +
      `(${totals.accepted} accepted; ${statusText}; ${totals.rejects} grip rejects); false starts ${totals.falseStarts}; ` +
      `simulated behaviours: ${behaviorText}; ` +
      `${totals.files} files, ${mb(totals.bytes)} (${mb(totals.bytes / Math.max(1, o.participants))} per session); ${seconds.toFixed(1)} s`,
  );
  console.log(
    `median attempt length (stop_ms, complete attempts): ` +
      SPEEDS.filter((sp) => lengths[sp].length > 0)
        .map((sp) => `${sp} ${(median(lengths[sp]) / 1000).toFixed(2)} s (n ${lengths[sp].length})`)
        .join(', ') +
      `; median session ${median(minutes).toFixed(1)} min (first trial to Finish)`,
  );
  console.log(
    `return of the arm: onset recorded in ${ret.onset}/${ret.complete} complete attempts (${pct(ret.onset, ret.complete)}), ` +
      `settled in ${ret.end} (${pct(ret.end, ret.complete)}); ${ret.lingered} lingered past return_cap_ms on purpose ` +
      `(late-return rate); seen in ${ret.onset}/${ret.complete - ret.lingered} of the others (${pct(ret.onset, ret.complete - ret.lingered)})`,
  );
  if (validate) {
    console.log(
      totals.invalid === 0
        ? `validation: all ${totals.files} records valid against schema/records.schema.json`
        : `validation: ${totals.invalid} of ${totals.files} records INVALID`,
    );
  }
  return totals.invalid > 0 ? 1 : 0;
}

main().then(
  (code) => process.exit(code),
  (e: unknown) => {
    console.error('synth failed:', e);
    process.exit(1);
  },
);
