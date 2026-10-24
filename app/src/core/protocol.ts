/**
 * Builds the session plan (blocks of trials) from a preset, with seeded randomization and
 * counterbalancing. Deterministic for a given (preset, seed, posture, firstHand, custom).
 *
 * PRESETS (PROTOCOL_VERSION 2.0.0). No preset asks for a screen-away grip (*_away): participants could
 * not do them; analyses synthesise screen-away data by rotating the sensor frame. A "main" block =
 * one hand × one orientation.
 *  street   (the default; 13 blocks, 39 trials, of which 29 in the 9 non-optional blocks; estimateMinutes
 *           5 min): the participant visit, most important conditions first, see STREET below.
 *  quick    (18 trials, 6 blocks; 3.5 min): orientations upright_toward, top_left_toward,
 *           top_right_toward; per block one trial each of speed natural, fast, slow; start 'natural';
 *           variant 'standard'.
 *  standard (48 trials, 10 blocks; 8 min): CORE_ORIENTATIONS (4); per block 4 trials:
 *           {natural, shoulder}, {natural, center}, {fast, natural}, {slow, natural};
 *           plus one 'odd' block per hand (orientation upright_toward, start natural, speed natural)
 *           with variants curved, pause, twist, bent_start, chin, lean_in, elbow_high, jerky (8).
 *  full     (86 trials, 14 blocks; 15 min): the 5 collected grips (COLLECTED_ORIENTATIONS without
 *           'free'); per block 6 trials: {natural,shoulder}, {natural,center}, {fast,shoulder},
 *           {fast,center}, {slow,shoulder}, {slow,center} (= 60); plus one odd block per hand with ALL 10
 *           ODD_VARIANTS (= 20); plus one 'free' block per hand (orientation 'free', speeds natural,
 *           fast, slow; start natural) (= 6).
 *  custom   : CustomProtocolSpec — main blocks = orientations × hands, each with speeds ×
 *           start_positions × reps trials; optional odd block per hand with odd_variants (one trial
 *           each; orientation upright_toward, speed natural, start natural, like the presets);
 *           optional free block per hand (natural, fast, slow). The spec is first passed through
 *           normalizeCustomSpec (see there), so buildPlan(spec) ≡ buildPlan(normalizeCustomSpec(spec)).
 *  ({speed, start_position} pairs; variant 'standard' unless stated; posture from the options.)
 *
 * STREET (A = first hand, B = the other; S = first sideways grip, S' = the other; U = upright_toward):
 *   block  tier  kind    hand  orientation  trials
 *     0     0    choice  A*    free         natural (the participant picks the hand and reports it after the trial)
 *     1     1    main    A     U            natural, natural, fast, slow
 *     2     1    main    B     U            natural, natural, fast, slow
 *     3     2    main    B     S            natural, natural, fast
 *     4     2    main    A     S            natural, natural, fast
 *     5     2    main    A     S'           natural, natural, fast
 *     6     2    main    B     S'           natural, natural, fast
 *     7     3    odd     B     U            the 4 variety cells (natural speed)
 *     8     3    odd     A     U            the same 4 cells (own shuffle)
 *     9     4+   main    A     U            natural, fast, natural
 *    10     4+   main    B     U            natural, fast, natural
 *    11     4+   odd     B     U            the 2 bonus variety cells
 *    12     4+   odd     A     U            the same 2 cells
 *   (* the planned hand of the choice block is a placeholder; + optional: true, run only when the
 *   participant opts in after tier 3.) Every street block carries `tier` and `optional`; the other
 *   presets set neither.
 *   S = top_left_toward when seed bit 1 is 0, top_right_toward when it is 1 (bit 0 picks A, as for every
 *   preset), so seeds 4k..4k+3 cover the four (first hand, first side) combinations.
 *   Variety cells: tilted_touch always, plus 3 cells drawn from STREET_VARIETY_POOL (start shoulder,
 *   start center, variants curved, pause, twist, elbow_high, bent_start, lean_in, jerky; all at natural
 *   speed, every other field at its default). The 2 bonus cells are the next 2 of the same draw, so they
 *   never repeat a tier-3 cell. chin and forehead never occur in street.
 *   Hands are balanced at every tier boundary; the hand switches (blocks 1→2, 3→4, 5→6, 7→8) alternate
 *   A→B, B→A, A→B, B→A.
 * hand_source: every planned condition says 'instructed', except the choice trial ('reported').
 *
 * ORDERING RULES:
 *  - firstHand defaults to (seed & 1) ? 'left' : 'right' (balances across participants).
 *  - quick / standard / full / custom: hands alternate in an ABBA pattern over orientation pairs: with
 *    orientations o1..on in a seeded random order, blocks are: A·o1, B·o1, B·o2, A·o2, A·o3, B·o3, ...
 *    (A = firstHand). The first orientation shown is ALWAYS upright_toward when present (easiest, used
 *    for familiarization). Odd blocks come after all main blocks (A then B); free blocks last (A then B).
 *  - street: the fixed block sequence above.
 *  - Trials inside a block are shuffled with the seeded RNG; `rep` counts repeats of the identical
 *    condition within the block, in presentation order (the first occurrence is rep 0).
 *    trial_index is global and contiguous from 0; block_index and trial_in_block are contiguous too.
 *  - Randomness: ONE stream, makeRng(mixSeed(seed, PLAN_SALT)), consumed first by the orientation order
 *    (street: by the variety draw, one shuffle of STREET_VARIETY_POOL), then by each block's shuffle in
 *    block order (a one-trial block draws nothing). The salt keeps the plan's stream distinct from any
 *    other use of the session seed. firstHand, posture and the street sideways side do not touch the
 *    stream, so overriding firstHand only swaps the hands of the same plan.
 *
 * IMPLEMENTED BY: protocol agent.
 */
import { COLLECTED_ORIENTATIONS, CORE_ORIENTATIONS, ODD_VARIANTS } from './constants';
import type {
  BlockKind,
  CustomProtocolSpec,
  Hand,
  OrientationId,
  PlannedBlock,
  PlannedTrial,
  Posture,
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
export interface PlanCell {
  speed: Speed;
  start_position: StartPosition;
  variant: Variant;
}

interface Design {
  orientations: readonly OrientationId[]; // main-block orientations (order is randomized later)
  main: readonly PlanCell[]; // the trials of every main block (repeated cells for reps > 1)
  odd: readonly Variant[]; // variants of the odd block; empty → no odd blocks
  free: boolean;
}

function cell(speed: Speed, start_position: StartPosition, variant: Variant = 'standard'): PlanCell {
  return { speed, start_position, variant };
}

type DesignPreset = Exclude<ProtocolPreset, 'custom' | 'street'>;

const PRESET_DESIGNS: Record<DesignPreset, Design> = {
  quick: {
    orientations: ['upright_toward', 'top_left_toward', 'top_right_toward'],
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
    orientations: COLLECTED_ORIENTATIONS.filter((o) => o !== 'free'),
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

/** The street preset's variety pool (tier 3 and the bonus): one non-default start or variant per cell. */
export const STREET_VARIETY_POOL: readonly PlanCell[] = [
  cell('natural', 'shoulder'),
  cell('natural', 'center'),
  cell('natural', 'natural', 'curved'),
  cell('natural', 'natural', 'pause'),
  cell('natural', 'natural', 'twist'),
  cell('natural', 'natural', 'elbow_high'),
  cell('natural', 'natural', 'bent_start'),
  cell('natural', 'natural', 'lean_in'),
  cell('natural', 'natural', 'jerky'),
];
/** Always one of the street preset's tier-3 variety cells. */
export const STREET_ALWAYS_CELL: PlanCell = cell('natural', 'natural', 'tilted_touch');
/** Variety cells drawn from STREET_VARIETY_POOL for tier 3 (plus STREET_ALWAYS_CELL) and for the bonus. */
const STREET_TIER3_DRAWN = 3;
const STREET_BONUS_DRAWN = 2;

export const DEFAULT_CUSTOM_SPEC: CustomProtocolSpec = {
  orientations: ['upright_toward', 'top_left_toward', 'top_right_toward'],
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
 * depends only on the chosen SETS); orientations are limited to COLLECTED_ORIENTATIONS (a screen-away
 * grip is dropped like an unknown value); an empty orientations/speeds/start_positions list falls back
 * to DEFAULT_CUSTOM_SPEC's; reps floored and clamped to 1..MAX_CUSTOM_REPS (non-numeric → 1);
 * include_free must be exactly true. sessionFlow should store this normalized spec in
 * session.protocol.custom so the stored spec describes the plan exactly.
 */
export function normalizeCustomSpec(spec: CustomProtocolSpec | null | undefined): CustomProtocolSpec {
  const s: Partial<CustomProtocolSpec> = spec ?? DEFAULT_CUSTOM_SPEC;
  const reps = Math.floor(Number(s.reps));
  return {
    orientations: pickAllowed(s.orientations, COLLECTED_ORIENTATIONS, DEFAULT_CUSTOM_SPEC.orientations),
    speeds: pickAllowed(s.speeds, SPEEDS, DEFAULT_CUSTOM_SPEC.speeds),
    start_positions: pickAllowed(s.start_positions, START_POSITIONS, DEFAULT_CUSTOM_SPEC.start_positions),
    reps: Number.isFinite(reps) ? Math.min(MAX_CUSTOM_REPS, Math.max(1, reps)) : 1,
    odd_variants: pickAllowed(s.odd_variants, VARIANTS, []),
    include_free: s.include_free === true,
  };
}

function customDesign(spec: CustomProtocolSpec | null | undefined): Design {
  const s = normalizeCustomSpec(spec);
  const main: PlanCell[] = [];
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
  kind: BlockKind;
  hand: Hand;
  orientation: OrientationId;
  cells: readonly PlanCell[];
  tier?: number;
  optional?: boolean;
}

/** quick / standard / full / custom: ABBA main blocks, then odd blocks, then free blocks. */
function designBlocks(design: Design, a: Hand, rng: () => number): BlockSpec[] {
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
  return specs;
}

/** The street preset's first sideways grip S and the other one S' (seed bit 1; never the RNG). */
function streetSideways(seed: number): [OrientationId, OrientationId] {
  // `>>>` works on the uint32 bit pattern, like `& 1` in resolveFirstHand
  return (seed >>> 1) & 1 ? ['top_right_toward', 'top_left_toward'] : ['top_left_toward', 'top_right_toward'];
}

/** The street preset (see STREET in the module doc). Consumes the RNG for the variety draw only. */
function streetBlocks(seed: number, a: Hand, rng: () => number): BlockSpec[] {
  const b = otherHand(a);
  const [s1, s2] = streetSideways(seed);
  const drawn = shuffle(STREET_VARIETY_POOL, rng);
  const variety = [STREET_ALWAYS_CELL, ...drawn.slice(0, STREET_TIER3_DRAWN)];
  const bonusVariety = drawn.slice(STREET_TIER3_DRAWN, STREET_TIER3_DRAWN + STREET_BONUS_DRAWN);
  const n = cell('natural', 'natural');
  const f = cell('fast', 'natural');
  const s = cell('slow', 'natural');
  const U: OrientationId = 'upright_toward';
  const block = (tier: number, kind: BlockKind, hand: Hand, orientation: OrientationId, cells: readonly PlanCell[]): BlockSpec => ({
    kind,
    hand,
    orientation,
    cells,
    tier,
    optional: tier === 4,
  });
  return [
    block(0, 'choice', a, 'free', [n]),
    block(1, 'main', a, U, [n, n, f, s]),
    block(1, 'main', b, U, [n, n, f, s]),
    block(2, 'main', b, s1, [n, n, f]),
    block(2, 'main', a, s1, [n, n, f]),
    block(2, 'main', a, s2, [n, n, f]),
    block(2, 'main', b, s2, [n, n, f]),
    block(3, 'odd', b, U, variety),
    block(3, 'odd', a, U, variety),
    block(4, 'main', a, U, [n, f, n]),
    block(4, 'main', b, U, [n, f, n]),
    block(4, 'odd', b, U, bonusVariety),
    block(4, 'odd', a, U, bonusVariety),
  ];
}

/** Shuffles each block's cells (in block order) and numbers the trials. */
function materialize(specs: readonly BlockSpec[], posture: Posture, rng: () => number): PlannedBlock[] {
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
        posture,
        rep,
        hand_source: spec.kind === 'choice' ? 'reported' : 'instructed',
      };
      return { trial_index: trialIndex++, block_index: blockIndex, trial_in_block: trialInBlock, condition };
    });
    return {
      block_index: blockIndex,
      kind: spec.kind,
      ...(spec.tier !== undefined ? { tier: spec.tier } : {}),
      ...(spec.optional !== undefined ? { optional: spec.optional } : {}),
      hand: spec.hand,
      orientation: spec.orientation,
      trials,
    };
  });
}

export function buildPlan(opts: ProtocolOptions): PlannedBlock[] {
  const preset: string = opts.preset;
  let design: Design | null = null;
  if (preset === 'custom') design = customDesign(opts.custom);
  else if (Object.prototype.hasOwnProperty.call(PRESET_DESIGNS, preset)) design = PRESET_DESIGNS[preset as DesignPreset];
  else if (preset !== 'street') throw new Error(`buildPlan: unknown preset "${String(opts.preset)}"`);
  const rng = makeRng(mixSeed(opts.seed, PLAN_SALT));
  const a = resolveFirstHand(opts);
  const specs = design ? designBlocks(design, a, rng) : streetBlocks(opts.seed, a, rng);
  return materialize(specs, opts.posture, rng);
}

/**
 * All trials in presentation order (the same objects, not copies). For a plan from buildPlan,
 * result[i].trial_index === i, so session.progress.next_trial_index indexes it directly.
 */
export function flattenPlan(plan: PlannedBlock[]): PlannedTrial[] {
  return plan.flatMap((b) => b.trials);
}

/** A bonus block (PlannedBlock.optional): run only when the participant opts in; a session is complete without it. */
export function isOptionalBlock(block: PlannedBlock): boolean {
  return block.optional === true;
}

/** Seconds per trial in the hands-free flow (recording + automatic advance), by speed. */
const TRIAL_SECONDS: Record<Speed, number> = { fast: 5.5, natural: 6.5, slow: 10 };
/** Extra seconds for a trial with a non-default start position or variant (an instruction to read). */
const READ_SECONDS = 2;
/** One tap on the next block's intro (hand / grip change). */
const BLOCK_CHANGE_SECONDS = 6;
/** Onboarding up to the first block. */
const SETUP_SECONDS = 40;

/**
 * SETUP 40 s + Σ per trial (fast 5.5 s, natural 6.5 s, slow 10 s; +2 s when the start position or the
 * variant is not the default) + 6 s per block change, over the non-optional, non-empty blocks, rounded
 * to 0.5 min. Block changes = (number of those blocks − 1). Presets: street 5 (its 4 bonus blocks
 * excluded), quick 3.5, standard 8, full 15 minutes.
 */
export function estimateMinutes(plan: PlannedBlock[]): number {
  const blocks = plan.filter((b) => !isOptionalBlock(b) && b.trials.length > 0);
  let seconds = SETUP_SECONDS + Math.max(0, blocks.length - 1) * BLOCK_CHANGE_SECONDS;
  for (const block of blocks) {
    for (const { condition: c } of block.trials) {
      const read = c.start_position !== 'natural' || c.variant !== 'standard' ? READ_SECONDS : 0;
      seconds += (TRIAL_SECONDS[c.speed] ?? TRIAL_SECONDS.natural) + read;
    }
  }
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

// Mirrored in analysis/motionlab/conventions.py (_SPEED_OVERRIDES / _VARIANT_OVERRIDES; a drift test
// parses these two objects).
const SPEED_OVERRIDES: Record<Speed, ParamOverride> = {
  natural: {},
  fast: { onset_sustain_ms: 40 },
  slow: {
    // floors stay above the still thresholds (20 / 0.5): at or below them, every loss of stillness
    // before GO counted as a false start (7 of the 8 false starts of the pilot)
    onset_gyro_min_dps: 25,
    onset_acc_min_ms2: 0.55,
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
 *  speed 'slow'   → onset_gyro_min_dps 25, onset_acc_min_ms2 0.55, end_gyro_dps 10, end_acc_ms2 0.25,
 *                   end_sustain_ms 450, max_movement_ms 12000
 *  speed 'fast'   → onset_sustain_ms 40
 *  variant 'pause'→ end_sustain_ms 1800, max_movement_ms 12000
 *  variant 'jerky'→ end_sustain_ms 600
 *  variant 'lean_in' / 'bent_start' / 'tilted_touch' / others → no change
 * Overrides compose (slow + pause → pause's end_sustain_ms wins, max of max_movement_ms).
 * An override only ever moves a parameter in the direction it exists for (thresholds and
 * onset_sustain_ms down, end_sustain_ms and max_movement_ms up). With the default base this gives
 * exactly the values above; with a researcher-tuned base it never undoes a tuning that already goes
 * further (e.g. a base max_movement_ms of 20000 stays 20000 for slow trials). Composition is
 * therefore "most extreme wins", which yields the rules above. Parameters that no override names
 * (e.g. return_cap_ms) are copied unchanged. Returns a new object; `base` is never mutated.
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

/** Number of planned trials of a preset, optional (bonus) blocks included; independent of the seed. */
export function presetTrialCount(preset: Exclude<ProtocolPreset, 'custom'>): number {
  // Trial counts do not depend on the seed; build once instead of duplicating the arithmetic.
  return flattenPlan(buildPlan({ preset, seed: 0, posture: 'standing' })).length;
}
