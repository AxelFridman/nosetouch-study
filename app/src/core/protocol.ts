/**
 * Builds the session plan (blocks of trials) from a preset, with seeded randomization and
 * counterbalancing. Deterministic for a given (preset, seed, posture, firstHand, custom).
 *
 * PRESETS (PROTOCOL_VERSION 1.0.0) — a "main" block = one hand × one orientation:
 *  quick    (24 trials, 8 blocks; estimateMinutes = 7.5 min): orientations upright_toward,
 *           upright_away, top_left_toward, top_right_toward; per block one trial each of speed
 *           natural, fast, slow; start 'natural'; variant 'standard'.
 *  standard (56 trials, 12 blocks; 14 min): CORE_ORIENTATIONS (5); per block 4 trials:
 *           {natural, shoulder}, {natural, center}, {fast, natural}, {slow, natural};
 *           plus one 'odd' block per hand (orientation upright_toward, start natural, speed natural)
 *           with variants curved, pause, twist, bent_start, chin, lean_in, elbow_high, jerky (8).
 *  full     (132 trials, 22 blocks; 29.5 min): all 9 orientations except 'free'; per block 6 trials:
 *           {natural,shoulder}, {natural,center}, {fast,shoulder}, {fast,center}, {slow,shoulder},
 *           {slow,center} (= 108); plus one odd block per hand with ALL 9 ODD_VARIANTS (= 18); plus
 *           one 'free' block per hand (orientation 'free', speeds natural, fast, slow; start natural) (= 6).
 *  custom   : CustomProtocolSpec — main blocks = orientations × hands, each with speeds ×
 *           start_positions × reps trials; optional odd block per hand with odd_variants (one trial
 *           each; orientation upright_toward, speed natural, start natural, like the presets);
 *           optional free block per hand (natural, fast, slow). The spec is first passed through
 *           normalizeCustomSpec (see there), so buildPlan(spec) ≡ buildPlan(normalizeCustomSpec(spec)).
 *  ({speed, start_position} pairs; variant 'standard' unless stated; posture from the options.)
 * ORDERING RULES:
 *  - firstHand defaults to (seed & 1) ? 'left' : 'right' (balances across participants).
 *  - Hands alternate in an ABBA pattern over orientation pairs: with orientations o1..on in a seeded
 *    random order, blocks are: A·o1, B·o1, B·o2, A·o2, A·o3, B·o3, ... (A = firstHand). The first
 *    orientation shown is ALWAYS upright_toward when present (easiest, used for familiarization).
 *  - Odd blocks come after all main blocks (A then B); free blocks last (A then B).
 *  - Trials inside a block are shuffled with the seeded RNG; `rep` counts repeats of the identical
 *    condition within the block, in presentation order (the first occurrence is rep 0).
 *    trial_index is global and contiguous from 0; block_index and trial_in_block are contiguous too.
 *  - Randomness: ONE stream, makeRng(mixSeed(seed, PLAN_SALT)), consumed by the orientation order
 *    first, then by each block's shuffle in block order. The salt keeps the plan's stream distinct
 *    from any other use of the session seed. firstHand and posture do not touch the stream, so
 *    overriding firstHand only swaps the hands of the same plan.
 *
 * IMPLEMENTED BY: protocol agent.
 */
import { ALL_ORIENTATIONS, CORE_ORIENTATIONS, ODD_VARIANTS } from './constants';
import type {
  CustomProtocolSpec,
  Hand,
  OrientationId,
  PlannedBlock,
  PlannedTrial,
  ProtocolOptions,
  ProtocolPreset,
  SegmentationParams,
  Speed,
  StartPosition,
  TrialCondition,
  Variant,
} from './types';
import { makeRng, mixSeed, shuffle } from './util';

/** Canonical orders of the condition enums (also used to order a normalized custom spec). */
export const SPEEDS: readonly Speed[] = ['natural', 'fast', 'slow'];
export const START_POSITIONS: readonly StartPosition[] = ['natural', 'shoulder', 'center'];
export const VARIANTS: readonly Variant[] = ['standard', ...ODD_VARIANTS];

/** Upper bound of CustomProtocolSpec.reps (schema maximum). */
export const MAX_CUSTOM_REPS = 10;

/** "PLAN" in ASCII: salt of the plan's random stream (see ORDERING RULES). */
const PLAN_SALT = 0x504c414e;
const FIRST_ORIENTATION: OrientationId = 'upright_toward';
const ODD_ORIENTATION: OrientationId = 'upright_toward';
const FREE_SPEEDS: readonly Speed[] = ['natural', 'fast', 'slow'];

/** One trial of a block before shuffling; hand, orientation and posture come from the block. */
interface Cell {
  speed: Speed;
  start_position: StartPosition;
  variant: Variant;
}

interface Design {
  orientations: readonly OrientationId[]; // main-block orientations (order is randomized later)
  main: readonly Cell[]; // the trials of every main block (repeated cells for reps > 1)
  odd: readonly Variant[]; // variants of the odd block; empty → no odd blocks
  free: boolean;
}

function cell(speed: Speed, start_position: StartPosition, variant: Variant = 'standard'): Cell {
  return { speed, start_position, variant };
}

const PRESET_DESIGNS: Record<Exclude<ProtocolPreset, 'custom'>, Design> = {
  quick: {
    orientations: ['upright_toward', 'upright_away', 'top_left_toward', 'top_right_toward'],
    main: [cell('natural', 'natural'), cell('fast', 'natural'), cell('slow', 'natural')],
    odd: [],
    free: false,
  },
  standard: {
    orientations: [...CORE_ORIENTATIONS],
    main: [cell('natural', 'shoulder'), cell('natural', 'center'), cell('fast', 'natural'), cell('slow', 'natural')],
    odd: ['curved', 'pause', 'twist', 'bent_start', 'chin', 'lean_in', 'elbow_high', 'jerky'],
    free: false,
  },
  full: {
    orientations: ALL_ORIENTATIONS.filter((o) => o !== 'free'),
    main: [
      cell('natural', 'shoulder'),
      cell('natural', 'center'),
      cell('fast', 'shoulder'),
      cell('fast', 'center'),
      cell('slow', 'shoulder'),
      cell('slow', 'center'),
    ],
    odd: [...ODD_VARIANTS],
    free: true,
  },
};

export const DEFAULT_CUSTOM_SPEC: CustomProtocolSpec = {
  orientations: ['upright_toward', 'upright_away'],
  speeds: ['natural', 'fast', 'slow'],
  start_positions: ['natural'],
  reps: 1,
  odd_variants: [],
  include_free: false,
};

/** Keeps the allowed values of `values` (deduplicated, in canonical order); empty → `fallback`. */
function pickAllowed<T extends string>(values: unknown, allowed: readonly T[], fallback: readonly T[]): T[] {
  const wanted = new Set<unknown>(Array.isArray(values) ? values : []);
  const out = allowed.filter((v) => wanted.has(v));
  return out.length > 0 ? out : [...fallback];
}

/**
 * Makes any custom spec (UI input, an old record, null) safe to plan AND schema-valid:
 * unknown values dropped, duplicates removed, lists put in canonical enum order (so the plan
 * depends only on the chosen SETS); an empty orientations/speeds/start_positions list falls back
 * to DEFAULT_CUSTOM_SPEC's; reps floored and clamped to 1..MAX_CUSTOM_REPS (non-numeric → 1);
 * include_free must be exactly true. sessionFlow should store this normalized spec in
 * session.protocol.custom so the stored spec describes the plan exactly.
 */
export function normalizeCustomSpec(spec: CustomProtocolSpec | null | undefined): CustomProtocolSpec {
  const s: Partial<CustomProtocolSpec> = spec ?? DEFAULT_CUSTOM_SPEC;
  const reps = Math.floor(Number(s.reps));
  return {
    orientations: pickAllowed(s.orientations, ALL_ORIENTATIONS, DEFAULT_CUSTOM_SPEC.orientations),
    speeds: pickAllowed(s.speeds, SPEEDS, DEFAULT_CUSTOM_SPEC.speeds),
    start_positions: pickAllowed(s.start_positions, START_POSITIONS, DEFAULT_CUSTOM_SPEC.start_positions),
    reps: Number.isFinite(reps) ? Math.min(MAX_CUSTOM_REPS, Math.max(1, reps)) : 1,
    odd_variants: pickAllowed(s.odd_variants, VARIANTS, []),
    include_free: s.include_free === true,
  };
}

function customDesign(spec: CustomProtocolSpec | null | undefined): Design {
  const s = normalizeCustomSpec(spec);
  const main: Cell[] = [];
  for (const speed of s.speeds) {
    for (const start of s.start_positions) {
      for (let r = 0; r < s.reps; r++) main.push(cell(speed, start));
    }
  }
  return { orientations: s.orientations, main, odd: s.odd_variants, free: s.include_free };
}

function otherHand(h: Hand): Hand {
  return h === 'left' ? 'right' : 'left';
}

/** upright_toward first (when present), the rest in seeded random order. */
function orientationOrder(orientations: readonly OrientationId[], rng: () => number): OrientationId[] {
  const rest = shuffle(
    orientations.filter((o) => o !== FIRST_ORIENTATION),
    rng,
  );
  return orientations.includes(FIRST_ORIENTATION) ? [FIRST_ORIENTATION, ...rest] : rest;
}

interface BlockSpec {
  kind: PlannedBlock['kind'];
  hand: Hand;
  orientation: OrientationId;
  cells: readonly Cell[];
}

export function buildPlan(opts: ProtocolOptions): PlannedBlock[] {
  const design = opts.preset === 'custom' ? customDesign(opts.custom) : PRESET_DESIGNS[opts.preset];
  if (!design) throw new Error(`buildPlan: unknown preset "${String(opts.preset)}"`);
  const rng = makeRng(mixSeed(opts.seed, PLAN_SALT));
  const a = resolveFirstHand(opts);
  const b = otherHand(a);

  const specs: BlockSpec[] = [];
  orientationOrder(design.orientations, rng).forEach((orientation, i) => {
    // ABBA: even orientation pairs start with A, odd ones with B
    for (const hand of i % 2 === 0 ? [a, b] : [b, a]) specs.push({ kind: 'main', hand, orientation, cells: design.main });
  });
  if (design.odd.length > 0) {
    const cells = design.odd.map((v) => cell('natural', 'natural', v));
    for (const hand of [a, b]) specs.push({ kind: 'odd', hand, orientation: ODD_ORIENTATION, cells });
  }
  if (design.free) {
    const cells = FREE_SPEEDS.map((speed) => cell(speed, 'natural'));
    for (const hand of [a, b]) specs.push({ kind: 'free', hand, orientation: 'free', cells });
  }

  let trialIndex = 0;
  return specs.map((spec, blockIndex): PlannedBlock => {
    const seen = new Map<string, number>();
    const trials = shuffle(spec.cells, rng).map((c, trialInBlock): PlannedTrial => {
      const key = `${c.speed}|${c.start_position}|${c.variant}`;
      const rep = seen.get(key) ?? 0;
      seen.set(key, rep + 1);
      const condition: TrialCondition = {
        hand: spec.hand,
        orientation: spec.orientation,
        speed: c.speed,
        start_position: c.start_position,
        variant: c.variant,
        posture: opts.posture,
        rep,
      };
      return { trial_index: trialIndex++, block_index: blockIndex, trial_in_block: trialInBlock, condition };
    });
    return { block_index: blockIndex, kind: spec.kind, hand: spec.hand, orientation: spec.orientation, trials };
  });
}

/**
 * All trials in presentation order (the same objects, not copies). For a plan from buildPlan,
 * result[i].trial_index === i, so session.progress.next_trial_index indexes it directly.
 */
export function flattenPlan(plan: PlannedBlock[]): PlannedTrial[] {
  return plan.flatMap((b) => b.trials);
}

const SECONDS_PER_TRIAL = 9;
const SECONDS_PER_BLOCK_CHANGE = 25;
const SETUP_SECONDS = 60;

/**
 * ≈ 9 s per trial + 25 s per block change + 60 s setup, rounded to 0.5 min.
 * Block changes = (number of non-empty blocks − 1); the first block's instructions are part of
 * the setup. Presets: quick 7.5, standard 14, full 29.5 minutes.
 */
export function estimateMinutes(plan: PlannedBlock[]): number {
  const blocks = plan.filter((b) => b.trials.length > 0);
  const nTrials = blocks.reduce((n, b) => n + b.trials.length, 0);
  const seconds =
    SETUP_SECONDS + nTrials * SECONDS_PER_TRIAL + Math.max(0, blocks.length - 1) * SECONDS_PER_BLOCK_CHANGE;
  return Math.round((seconds / 60) * 2) / 2;
}

/** Resolved first hand for these options (see ordering rules). */
export function resolveFirstHand(opts: ProtocolOptions): 'left' | 'right' {
  if (opts.firstHand === 'left' || opts.firstHand === 'right') return opts.firstHand;
  // `& 1` works on the uint32 bit pattern, so seeds ≥ 2^31 keep their parity
  return opts.seed & 1 ? 'left' : 'right';
}

/** Parameters that per-condition overrides may change, and the direction each may move. */
type TunedParam =
  | 'onset_gyro_min_dps'
  | 'onset_acc_min_ms2'
  | 'onset_sustain_ms'
  | 'end_gyro_dps'
  | 'end_acc_ms2'
  | 'end_sustain_ms'
  | 'max_movement_ms';
type ParamOverride = Partial<Record<TunedParam, number>>;

const TUNE_DIRECTION: Record<TunedParam, 'min' | 'max'> = {
  onset_gyro_min_dps: 'min', // slow movements: lower onset floors so onset is still detected
  onset_acc_min_ms2: 'min',
  onset_sustain_ms: 'min', // fast movements: confirm onset sooner
  end_gyro_dps: 'min', // slow movements: the hand can be slower than the default end threshold mid-movement
  end_acc_ms2: 'min',
  end_sustain_ms: 'max', // pauses / stop-go: a short stop must not count as the end
  max_movement_ms: 'max',
};

const SPEED_OVERRIDES: Record<Speed, ParamOverride> = {
  natural: {},
  fast: { onset_sustain_ms: 40 },
  slow: {
    onset_gyro_min_dps: 15,
    onset_acc_min_ms2: 0.3,
    end_gyro_dps: 10,
    end_acc_ms2: 0.25,
    end_sustain_ms: 450,
    max_movement_ms: 12000,
  },
};

const VARIANT_OVERRIDES: Partial<Record<Variant, ParamOverride>> = {
  pause: { end_sustain_ms: 1800, max_movement_ms: 12000 },
  jerky: { end_sustain_ms: 600 },
};

/**
 * Per-condition overrides of the base segmentation parameters:
 *  speed 'slow'   → onset_gyro_min_dps 15, onset_acc_min_ms2 0.3, end_gyro_dps 10, end_acc_ms2 0.25,
 *                   end_sustain_ms 450, max_movement_ms 12000
 *  speed 'fast'   → onset_sustain_ms 40
 *  variant 'pause'→ end_sustain_ms 1800, max_movement_ms 12000
 *  variant 'jerky'→ end_sustain_ms 600
 *  variant 'lean_in' / 'bent_start' / others → no change
 * Overrides compose (slow + pause → pause's end_sustain_ms wins, max of max_movement_ms).
 * An override only ever moves a parameter in the direction it exists for (thresholds and
 * onset_sustain_ms down, end_sustain_ms and max_movement_ms up). With the default base this gives
 * exactly the values above; with a researcher-tuned base it never undoes a tuning that already goes
 * further (e.g. a base max_movement_ms of 20000 stays 20000 for slow trials). Composition is
 * therefore "most extreme wins", which yields the rules above. Returns a new object; `base` is
 * never mutated.
 */
export function resolveParams(base: SegmentationParams, c: TrialCondition): SegmentationParams {
  const out: SegmentationParams = { ...base };
  for (const override of [SPEED_OVERRIDES[c.speed], VARIANT_OVERRIDES[c.variant]]) {
    if (!override) continue;
    for (const key of Object.keys(override) as TunedParam[]) {
      const value = override[key] as number;
      out[key] = TUNE_DIRECTION[key] === 'min' ? Math.min(out[key], value) : Math.max(out[key], value);
    }
  }
  return out;
}

export function presetTrialCount(preset: Exclude<ProtocolPreset, 'custom'>): number {
  // Trial counts do not depend on the seed; build once instead of duplicating the arithmetic.
  return flattenPlan(buildPlan({ preset, seed: 0, posture: 'standing' })).length;
}
