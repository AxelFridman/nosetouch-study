/**
 * Generates a synthetic dataset with the REAL recording pipeline (protocol → simulated participant →
 * TrialController → TrialRecord; see src/sim/dataset.ts) and writes it in the backend's folder layout:
 *   <out>/<study_code>/<participant_id>/<session_id>/session.json
 *   <out>/<study_code>/<participant_id>/<session_id>/trial_<idx:3>_a<attempt>_<trial_id>.json
 * Usage: npm run synth -- --participants 12 --preset standard --seed 42 --left-fraction 0.3 \
 *          --out ../analysis/data/synthetic [--study synthetic-v1] [--failure-rate 0.03] [--validate]
 *          [--ios-gravity-sign -1] [--clean]
 *   --participants N      simulated participants, one session each (default 12)
 *   --preset P            quick | standard | full | custom (custom = protocol.DEFAULT_CUSTOM_SPEC)
 *   --seed N              uint32 master seed (default 42): same seed → byte-identical files
 *   --left-fraction F     share of left-handed participants (default 0.3, rounded to whole people)
 *   --failure-rate F      share of attempts performed as 'no_move' / 'false_start' (default 0.03)
 *   --study CODE          study_code (default synthetic-v1); never a real study's code
 *   --out DIR             output root (default ../analysis/data/synthetic, relative to the cwd)
 *   --validate            validate every record with ajv against schema/records.schema.json
 *   --ios-gravity-sign S  acc/accg sign of the iOS-like devices: -1 (WebKit, default) or 1
 *   --clean               delete <out>/<study_code>/ first (otherwise other runs' files stay)
 * Every record has simulated: true. Platforms alternate iOS-like (WebKit: gravity sign −1, full
 * precision, no quaternion) and Android-like (Chrome: +1, values quantised to 0.1 units, quaternion);
 * profiles carry EHI answers consistent with the simulated handedness.
 * Prints a one-line summary per participant and the totals. Exit code 1 if a record fails validation.
 *
 * IMPLEMENTED BY: simulation agent.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ajv2020 } from 'ajv/dist/2020.js';
import type { ProtocolPreset, TrialStatus } from '../src/core/types';
import { iterateDataset, resolveDatasetOptions, summarizeSession, type DatasetOptions } from '../src/sim/dataset';

const FLAGS = new Set(['validate', 'clean', 'help']);
const VALUED = new Set(['participants', 'preset', 'seed', 'left-fraction', 'failure-rate', 'study', 'out', 'ios-gravity-sign']);

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

const STATUSES: TrialStatus[] = ['complete', 'no_movement', 'too_long', 'aborted', 'interrupted'];

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
      `failure rate ${o.failureRate}, study ${o.studyCode}, iOS gravity sign ${o.iosGravitySign} → ${studyDir}`,
  );
  const started = Date.now();
  const totals = { files: 0, bytes: 0, trials: 0, accepted: 0, falseStarts: 0, invalid: 0, completed: 0, abandoned: 0 };
  const statusTotals = Object.fromEntries(STATUSES.map((s) => [s, 0])) as Record<TrialStatus, number>;
  const behaviorTotals: Record<string, number> = {};
  const byPlatform = { ios: 0, android: 0 };
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
    if (sum.status === 'completed') totals.completed++;
    else totals.abandoned++;
    for (const st of STATUSES) statusTotals[st] += sum.statuses[st];
    for (const [b, n] of Object.entries(sum.behaviors)) behaviorTotals[b] = (behaviorTotals[b] ?? 0) + (n ?? 0);
    byPlatform[sum.platform]++;
    if (sum.dominant === 'left') nLeft++;
    const failed = STATUSES.filter((st) => st !== 'complete' && sum.statuses[st] > 0)
      .map((st) => `${st} ${sum.statuses[st]}`)
      .join(', ');
    console.log(
      `${String(s.design.index + 1).padStart(3)}/${o.participants} ${sum.participant_id} ${sum.platform.padEnd(7)} ` +
        `${sum.dominant === 'left' ? 'left ' : 'right'}-handed (EHI ${sum.ehi_class ?? '-'}) ${sum.sex} ${sum.age_years ?? '?'} y ` +
        `${sum.height_cm ?? '?'} cm, first ${sum.first_hand}, gravity sign ${sum.gravity_sign ?? '?'}: ` +
        `${sum.n_accepted}/${sum.n_planned} accepted in ${sum.n_attempts} attempts${failed ? ` (${failed})` : ''}, ` +
        `false starts ${sum.false_starts}, ${sum.status}, ${mb(bytes)}, ${((Date.now() - t1) / 1000).toFixed(1)} s`,
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
  console.log(
    `total: ${o.participants} participants (${byPlatform.ios} iOS-like, ${byPlatform.android} Android-like; ${nLeft} left-handed), ` +
      `${totals.completed} sessions completed, ${totals.abandoned} abandoned; ${totals.trials} trial records ` +
      `(${totals.accepted} accepted; ${statusText}); false starts ${totals.falseStarts}; simulated behaviours: ${behaviorText}; ` +
      `${totals.files} files, ${mb(totals.bytes)} (${mb(totals.bytes / Math.max(1, o.participants))} per session); ${seconds.toFixed(1)} s`,
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
