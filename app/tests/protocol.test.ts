import { Ajv2020 } from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';
import {
  ALL_ORIENTATIONS,
  COLLECTED_ORIENTATIONS,
  CORE_ORIENTATIONS,
  DEFAULT_SEGMENTATION_PARAMS,
  ODD_VARIANTS,
  ORIENTATIONS,
  PROTOCOL_VERSION,
} from '../src/core/constants';
import {
  DEFAULT_CUSTOM_SPEC,
  MAX_CUSTOM_REPS,
  SPEEDS,
  START_POSITIONS,
  STREET_ALWAYS_CELL,
  STREET_VARIETY_POOL,
  VARIANTS,
  buildPlan,
  estimateMinutes,
  flattenPlan,
  isOptionalBlock,
  normalizeCustomSpec,
  presetTrialCount,
  resolveFirstHand,
  resolveParams,
} from '../src/core/protocol';
import type {
  BlockKind,
  CustomProtocolSpec,
  Hand,
  OrientationId,
  PlannedBlock,
  Posture,
  ProtocolOptions,
  ProtocolPreset,
  SegmentationParams,
  Speed,
  StartPosition,
  TrialCondition,
  Variant,
} from '../src/core/types';
import { makeRng, shuffle } from '../src/core/util';
import { expectValid, recordsSchema } from './helpers/schema';

const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addSchema(recordsSchema);

/** Schema errors of `value` against any JSON pointer inside records.schema.json. */
function schemaErrors(pointer: string, value: unknown): string[] {
  const fn = ajv.getSchema(`${recordsSchema.$id}#${pointer}`);
  if (!fn) throw new Error(`no schema at ${pointer}`);
  return fn(value) ? [] : (fn.errors ?? []).map((e) => `${e.instancePath || '(root)'} ${e.message}`);
}

/** Presets built from a Design (ABBA over orientations, then odd and free blocks). */
const PRESETS = ['quick', 'standard', 'full'] as const;
/** Every fixed preset, street included. */
const FIXED_PRESETS = ['street', ...PRESETS] as const;
const SEEDS = [0, 1, 2, 3, 7, 42, 12345, 123456789, 2 ** 31, 2 ** 32 - 1];
const POSTURES: Posture[] = ['standing', 'sitting'];
const EXPECTED = {
  street: { trials: 39, blocks: 13, minutes: 5 },
  quick: { trials: 18, blocks: 6, minutes: 3.5 },
  standard: { trials: 48, blocks: 10, minutes: 8 },
  full: { trials: 86, blocks: 14, minutes: 15 },
} as const;
const NON_FREE = COLLECTED_ORIENTATIONS.filter((o) => o !== 'free');
const STANDARD_ODD: Variant[] = ['curved', 'pause', 'twist', 'bent_start', 'chin', 'lean_in', 'elbow_high', 'jerky'];
const U: OrientationId = 'upright_toward';

const other = (h: Hand): Hand => (h === 'left' ? 'right' : 'left');
const key = (...parts: string[]): string => parts.join('|');
const cellKey = (c: Pick<TrialCondition, 'speed' | 'start_position' | 'variant'>): string => key(c.speed, c.start_position, c.variant);
const mainBlocks = (plan: PlannedBlock[]): PlannedBlock[] => plan.filter((b) => b.kind === 'main');
const isAway = (o: OrientationId): boolean => ORIENTATIONS[o].screen === 'away';

/** Multiset of keys → counts. */
function countBy<T>(items: T[], f: (x: T) => string): Map<string, number> {
  const m = new Map<string, number>();
  for (const x of items) m.set(f(x), (m.get(f(x)) ?? 0) + 1);
  return m;
}

/** Sorted cell keys of a block (its multiset of conditions, ignoring the order). */
const cellsOf = (b: PlannedBlock): string[] => b.trials.map((t) => cellKey(t.condition)).sort();

/** Numbering, copies of the block fields, posture, rep and hand_source: shared by every preset. */
function expectNumbered(plan: PlannedBlock[], opts: ProtocolOptions): void {
  let next = 0;
  const conditions = new Set<TrialCondition>();
  plan.forEach((block, bi) => {
    expectValid('PlannedBlock', block);
    expect(block.block_index).toBe(bi);
    expect(block.trials.length).toBeGreaterThan(0);
    const reps = new Map<string, number>();
    block.trials.forEach((t, ti) => {
      expect(t.trial_index).toBe(next++);
      expect(t.block_index).toBe(bi);
      expect(t.trial_in_block).toBe(ti);
      const c = t.condition;
      expect(c.hand).toBe(block.hand);
      expect(c.orientation).toBe(block.orientation);
      expect(c.posture).toBe(opts.posture);
      expect(isAway(c.orientation)).toBe(false);
      // rep = number of earlier identical conditions in this block (presentation order)
      expect(c.rep).toBe(reps.get(cellKey(c)) ?? 0);
      reps.set(cellKey(c), (reps.get(cellKey(c)) ?? 0) + 1);
      expect(c.hand_source).toBe(block.kind === 'choice' ? 'reported' : 'instructed');
      expect(conditions.has(c)).toBe(false); // no shared condition objects
      conditions.add(c);
    });
  });
}

/** Every structural rule of the Design presets (see protocol.ts ORDERING RULES). */
function expectWellFormed(plan: PlannedBlock[], opts: ProtocolOptions): void {
  expectNumbered(plan, opts);
  const a = resolveFirstHand(opts);
  const b = other(a);
  const phase: Record<BlockKind, number> = { choice: -1, main: 0, odd: 1, free: 2 };
  let lastPhase = 0;
  for (const block of plan) {
    expect(block.kind).not.toBe('choice');
    expect(block).not.toHaveProperty('tier'); // tiers belong to the street preset only
    expect(block).not.toHaveProperty('optional');
    expect(phase[block.kind]).toBeGreaterThanOrEqual(lastPhase); // main → odd → free
    lastPhase = phase[block.kind];
    if (block.kind === 'odd') expect(block.orientation).toBe('upright_toward');
    if (block.kind === 'free') expect(block.orientation).toBe('free');
    for (const { condition: c } of block.trials) {
      if (block.kind !== 'odd') expect(c.variant).toBe('standard');
      if (block.kind !== 'main') expect(c.start_position).toBe('natural');
      if (block.kind === 'odd') expect(c.speed).toBe('natural');
    }
  }
  // ABBA over orientation pairs of the main blocks
  const main = mainBlocks(plan);
  expect(main.length % 2).toBe(0);
  for (let i = 0; i < main.length / 2; i++) {
    expect(main[2 * i].orientation).toBe(main[2 * i + 1].orientation);
    expect([main[2 * i].hand, main[2 * i + 1].hand]).toEqual(i % 2 === 0 ? [a, b] : [b, a]);
  }
  expect(new Set(main.map((m) => m.orientation)).size).toBe(main.length / 2); // each orientation = one pair
  for (const kind of ['odd', 'free'] as const) {
    const blocks = plan.filter((x) => x.kind === kind);
    if (blocks.length > 0) expect(blocks.map((x) => x.hand)).toEqual([a, b]);
  }
  if (main.some((m) => m.orientation === 'upright_toward')) expect(plan[0].orientation).toBe('upright_toward');
}

describe('buildPlan — presets', () => {
  it.each(FIXED_PRESETS)('%s has the documented trial and block counts for every seed', (preset) => {
    for (const seed of SEEDS) {
      const plan = buildPlan({ preset, seed, posture: 'standing' });
      expect(flattenPlan(plan)).toHaveLength(EXPECTED[preset].trials);
      expect(plan).toHaveLength(EXPECTED[preset].blocks);
    }
    expect(presetTrialCount(preset)).toBe(EXPECTED[preset].trials);
  });

  it('every block validates against the schema and all ordering rules hold', () => {
    for (const preset of PRESETS) {
      for (const seed of SEEDS) {
        for (const posture of POSTURES) {
          const opts: ProtocolOptions = { preset, seed, posture };
          expectWellFormed(buildPlan(opts), opts);
          for (const firstHand of ['left', 'right'] as Hand[]) {
            const withHand = { ...opts, firstHand };
            expectWellFormed(buildPlan(withHand), withHand);
          }
        }
      }
    }
  });

  it('no preset plan asks for a screen-away grip (seeds 0..199, default custom spec included)', () => {
    const presets: { preset: ProtocolPreset; custom?: CustomProtocolSpec }[] = [
      ...FIXED_PRESETS.map((preset) => ({ preset })),
      { preset: 'custom', custom: DEFAULT_CUSTOM_SPEC },
      { preset: 'custom' }, // a missing spec means DEFAULT_CUSTOM_SPEC
    ];
    for (const { preset, custom } of presets) {
      for (let seed = 0; seed < 200; seed++) {
        const plan = buildPlan({ preset, seed, posture: 'standing', custom });
        const away = plan.flatMap((b) => [b.orientation, ...b.trials.map((t) => t.condition.orientation)]).filter(isAway);
        expect(away, `${preset} seed ${seed}`).toEqual([]);
      }
    }
  });

  it('quick: 3 screen-toward orientations × 2 hands × {natural, fast, slow}, start natural, standard variant', () => {
    const plan = buildPlan({ preset: 'quick', seed: 9, posture: 'standing' });
    expect(plan.every((b) => b.kind === 'main')).toBe(true);
    const counts = countBy(flattenPlan(plan), (t) => key(t.condition.hand, t.condition.orientation, cellKey(t.condition)));
    expect(counts.size).toBe(18);
    for (const hand of ['left', 'right']) {
      for (const o of ['upright_toward', 'top_left_toward', 'top_right_toward']) {
        for (const speed of ['natural', 'fast', 'slow']) expect(counts.get(key(hand, o, speed, 'natural', 'standard'))).toBe(1);
      }
    }
  });

  it('standard: every (hand, orientation, speed, start_position) main condition exactly once; odd blocks hold the 8 listed variants per hand', () => {
    expect(CORE_ORIENTATIONS).toEqual(['upright_toward', 'top_left_toward', 'top_right_toward', 'upside_down_toward']);
    for (const seed of SEEDS) {
      const plan = buildPlan({ preset: 'standard', seed, posture: 'sitting' });
      const mainTrials = flattenPlan(mainBlocks(plan));
      const counts = countBy(mainTrials, (t) =>
        key(t.condition.hand, t.condition.orientation, t.condition.speed, t.condition.start_position),
      );
      expect(mainTrials).toHaveLength(32);
      expect(counts.size).toBe(32);
      const cells = [
        ['natural', 'shoulder'],
        ['natural', 'center'],
        ['fast', 'natural'],
        ['slow', 'natural'],
      ];
      for (const hand of ['left', 'right']) {
        for (const o of CORE_ORIENTATIONS) for (const [sp, st] of cells) expect(counts.get(key(hand, o, sp, st))).toBe(1);
      }
      const odd = plan.filter((b) => b.kind === 'odd');
      expect(odd).toHaveLength(2);
      expect(new Set(odd.map((b) => b.hand))).toEqual(new Set(['left', 'right']));
      for (const block of odd) {
        expect(block.trials.map((t) => t.condition.variant).sort()).toEqual([...STANDARD_ODD].sort());
      }
      expect(plan.some((b) => b.kind === 'free')).toBe(false);
    }
  });

  it('full: the 5 collected grips × 2 hands × 6 cells once each, all 10 odd variants per hand, free blocks last', () => {
    expect(NON_FREE).toEqual(['upright_toward', 'top_left_toward', 'top_right_toward', 'upside_down_toward', 'flat_up']);
    const plan = buildPlan({ preset: 'full', seed: 314, posture: 'standing' });
    const main = flattenPlan(mainBlocks(plan));
    const counts = countBy(main, (t) => key(t.condition.hand, t.condition.orientation, cellKey(t.condition)));
    expect(main).toHaveLength(60);
    expect(counts.size).toBe(60);
    for (const hand of ['left', 'right']) {
      for (const o of NON_FREE) {
        for (const sp of ['natural', 'fast', 'slow']) {
          for (const st of ['shoulder', 'center']) expect(counts.get(key(hand, o, sp, st, 'standard'))).toBe(1);
        }
      }
    }
    const odd = plan.filter((b) => b.kind === 'odd');
    expect(odd).toHaveLength(2);
    expect(ODD_VARIANTS).toHaveLength(10);
    for (const block of odd) expect(block.trials.map((t) => t.condition.variant).sort()).toEqual([...ODD_VARIANTS].sort());
    const free = plan.filter((b) => b.kind === 'free');
    expect(free).toHaveLength(2);
    expect(plan.slice(-2)).toEqual(free);
    for (const block of free) {
      expect(block.trials.map((t) => t.condition.speed).sort()).toEqual(['fast', 'natural', 'slow']);
    }
  });

  it('the first block is always upright_toward (familiarization)', () => {
    for (const preset of PRESETS) {
      for (let seed = 0; seed < 300; seed++) {
        expect(buildPlan({ preset, seed, posture: 'standing' })[0].orientation).toBe('upright_toward');
      }
    }
  });

  it('hands follow ABBA: A·o1, B·o1, B·o2, A·o2, A·o3, B·o3, ...', () => {
    const plan = buildPlan({ preset: 'standard', seed: 4, posture: 'standing' }); // even seed → right first
    const R = 'right';
    const L = 'left';
    expect(plan.map((b) => b.hand)).toEqual([R, L, L, R, R, L, L, R, R, L]);
    const swapped = buildPlan({ preset: 'standard', seed: 4, posture: 'standing', firstHand: 'left' });
    expect(swapped.map((b) => b.hand)).toEqual([L, R, R, L, L, R, R, L, L, R]);
  });

  it('pins the generated plan for one seed (a change here needs a PROTOCOL_VERSION bump)', () => {
    expect(PROTOCOL_VERSION).toBe('2.0.0');
    const plan = buildPlan({ preset: 'standard', seed: 12345, posture: 'standing' });
    expect(plan.map((b) => `${b.kind}:${b.hand}:${b.orientation}`)).toEqual([
      'main:left:upright_toward',
      'main:right:upright_toward',
      'main:right:top_left_toward',
      'main:left:top_left_toward',
      'main:left:upside_down_toward',
      'main:right:upside_down_toward',
      'main:right:top_right_toward',
      'main:left:top_right_toward',
      'odd:left:upright_toward',
      'odd:right:upright_toward',
    ]);
    expect(plan[0].trials.map((t) => `${t.condition.speed}/${t.condition.start_position}`)).toEqual([
      'fast/natural',
      'natural/center',
      'natural/shoulder',
      'slow/natural',
    ]);
    expect(plan[8].trials.map((t) => t.condition.variant)).toEqual([
      'lean_in',
      'bent_start',
      'elbow_high',
      'twist',
      'jerky',
      'pause',
      'chin',
      'curved',
    ]);
  });
});

// ───────────────────────────── street ─────────────────────────────

const STREET_TIERS = [0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4];
const STREET_KINDS: BlockKind[] = ['choice', 'main', 'main', 'main', 'main', 'main', 'main', 'odd', 'odd', 'main', 'main', 'odd', 'odd'];
const POOL_KEYS = new Set(STREET_VARIETY_POOL.map(cellKey));
const TILTED = cellKey(STREET_ALWAYS_CELL);
const street = (seed: number, extra: Partial<ProtocolOptions> = {}): PlannedBlock[] =>
  buildPlan({ preset: 'street', seed, posture: 'standing', ...extra });

/** First sideways grip S of the street preset, from seed bit 1 (independent of the code under test). */
const expectedSideways = (seed: number): OrientationId => (Math.floor(seed / 2) % 2 === 0 ? 'top_left_toward' : 'top_right_toward');
const otherSide = (o: OrientationId): OrientationId => (o === 'top_left_toward' ? 'top_right_toward' : 'top_left_toward');

describe('buildPlan — street (protocol 2.0.0)', () => {
  it('has the 13 blocks of spec section 2: tiers, kinds, optional bonus, hands, grips and trials', () => {
    for (let seed = 0; seed < 200; seed++) {
      for (const posture of POSTURES) {
        const opts: ProtocolOptions = { preset: 'street', seed, posture };
        const plan = buildPlan(opts);
        expectNumbered(plan, opts);
        const A = resolveFirstHand(opts);
        const B = other(A);
        const S = expectedSideways(seed);
        const S2 = otherSide(S);
        expect(plan.map((b) => b.tier)).toEqual(STREET_TIERS);
        expect(plan.map((b) => b.kind)).toEqual(STREET_KINDS);
        expect(plan.map((b) => b.optional)).toEqual(STREET_TIERS.map((t) => t === 4));
        expect(plan.map(isOptionalBlock)).toEqual(STREET_TIERS.map((t) => t === 4));
        expect(plan.map((b) => b.hand)).toEqual([A, A, B, B, A, A, B, B, A, A, B, B, A]);
        expect(plan.map((b) => b.orientation)).toEqual(['free', U, U, S, S, S2, S2, U, U, U, U, U, U]);

        const n = cellKey({ speed: 'natural', start_position: 'natural', variant: 'standard' });
        const f = cellKey({ speed: 'fast', start_position: 'natural', variant: 'standard' });
        const s = cellKey({ speed: 'slow', start_position: 'natural', variant: 'standard' });
        expect(cellsOf(plan[0])).toEqual([n]);
        for (const bi of [1, 2]) expect(cellsOf(plan[bi])).toEqual([n, n, f, s].sort());
        for (const bi of [3, 4, 5, 6]) expect(cellsOf(plan[bi])).toEqual([n, n, f].sort());
        for (const bi of [9, 10]) expect(cellsOf(plan[bi])).toEqual([n, n, f].sort());
        expect(plan[7].trials).toHaveLength(4);
        expect(plan[11].trials).toHaveLength(2);

        // the choice trial: natural / natural / standard, hand reported by the participant
        expect(plan[0].trials[0].condition).toMatchObject({ speed: 'natural', start_position: 'natural', variant: 'standard', hand_source: 'reported' });
        expect(flattenPlan(plan).filter((t) => t.condition.hand_source === 'reported')).toHaveLength(1);
      }
    }
  });

  it('variety: tilted_touch + 3 pool cells in tier 3 (same set for both hands), 2 other pool cells in the bonus; never chin or forehead', () => {
    for (let seed = 0; seed < 200; seed++) {
      const plan = street(seed);
      const tier3 = plan.filter((b) => b.tier === 3);
      const bonusOdd = plan.filter((b) => b.tier === 4 && b.kind === 'odd');
      expect(tier3).toHaveLength(2);
      expect(bonusOdd).toHaveLength(2);
      expect(cellsOf(tier3[0])).toEqual(cellsOf(tier3[1]));
      expect(cellsOf(bonusOdd[0])).toEqual(cellsOf(bonusOdd[1]));
      const t3 = cellsOf(tier3[0]);
      expect(t3).toContain(TILTED);
      expect(new Set(t3).size).toBe(4); // 4 different cells
      const drawn = t3.filter((k) => k !== TILTED);
      expect(drawn).toHaveLength(3);
      for (const k of drawn) expect(POOL_KEYS.has(k), k).toBe(true);
      const bonus = cellsOf(bonusOdd[0]);
      expect(new Set(bonus).size).toBe(2);
      for (const k of bonus) {
        expect(POOL_KEYS.has(k), k).toBe(true);
        expect(t3).not.toContain(k); // the bonus cells are disjoint from tier 3
      }
      for (const block of [...tier3, ...bonusOdd]) {
        for (const { condition: c } of block.trials) {
          expect(c.speed).toBe('natural');
          // one non-default field per cell
          expect(Number(c.start_position !== 'natural') + Number(c.variant !== 'standard')).toBe(1);
        }
      }
      const variants = flattenPlan(plan).map((t) => t.condition.variant);
      expect(variants).not.toContain('chin');
      expect(variants).not.toContain('forehead');
      expect(flattenPlan(plan).filter((t) => t.condition.variant === 'tilted_touch').map((t) => plan[t.block_index].tier)).toEqual([3, 3]);
    }
  });

  it('the variety draw depends on the seed and reaches every pool cell, in tier 3 and in the bonus', () => {
    const tier3Sets = new Set<string>();
    const inTier3 = new Set<string>();
    const inBonus = new Set<string>();
    for (let seed = 0; seed < 200; seed++) {
      const plan = street(seed);
      const t3 = cellsOf(plan[7]).filter((k) => k !== TILTED);
      tier3Sets.add(t3.join(','));
      t3.forEach((k) => inTier3.add(k));
      cellsOf(plan[11]).forEach((k) => inBonus.add(k));
    }
    expect(tier3Sets.size).toBeGreaterThan(40); // C(9, 3) = 84 possible sets
    expect(inTier3).toEqual(POOL_KEYS);
    expect(inBonus).toEqual(POOL_KEYS);
  });

  it('seed bit 0 picks the first hand and bit 1 the first sideways grip; seeds 0..3 cover all four combinations', () => {
    const combos = new Set<string>();
    for (let seed = 0; seed < 4; seed++) {
      const plan = street(seed);
      const A: Hand = seed & 1 ? 'left' : 'right';
      expect(plan[1].hand).toBe(A);
      expect(plan[3].orientation).toBe(seed & 2 ? 'top_right_toward' : 'top_left_toward');
      combos.add(`${plan[1].hand}|${plan[3].orientation}`);
    }
    expect(combos.size).toBe(4);
    for (const seed of SEEDS) {
      const plan = street(seed);
      expect(plan[1].hand).toBe(resolveFirstHand({ preset: 'street', seed, posture: 'standing' }));
      expect(plan[3].orientation).toBe(expectedSideways(seed));
      expect(plan[5].orientation).toBe(otherSide(expectedSideways(seed)));
    }
  });

  it('hands are balanced at every tier boundary and switch A→B, B→A, A→B, B→A in tiers 1–3', () => {
    for (const seed of SEEDS) {
      const plan = street(seed);
      const A = plan[1].hand;
      const count = { left: 0, right: 0 };
      for (let tier = 1; tier <= 4; tier++) {
        for (const b of plan.filter((x) => x.tier === tier)) count[b.hand] += b.trials.length;
        expect(count.left, `after tier ${tier}`).toBe(count.right);
      }
      const hands = plan.filter((b) => b.tier !== undefined && b.tier >= 1 && b.tier <= 3).map((b) => b.hand);
      const switches = hands.slice(1).flatMap((h, i) => (h !== hands[i] ? [`${hands[i]}→${h}`] : []));
      expect(switches).toEqual([`${A}→${other(A)}`, `${other(A)}→${A}`, `${A}→${other(A)}`, `${other(A)}→${A}`]);
    }
  });

  it('firstHand only swaps the hands; posture only changes the posture; neither touches the random draws', () => {
    for (const seed of [0, 1, 2, 3, 77, 12345]) {
      const base = street(seed, { firstHand: 'right' });
      const alt = street(seed, { firstHand: 'left', posture: 'sitting' });
      expect(alt.map((b) => [b.kind, b.tier, b.orientation, other(b.hand)])).toEqual(base.map((b) => [b.kind, b.tier, b.orientation, b.hand]));
      expect(flattenPlan(alt).map((t) => cellKey(t.condition))).toEqual(flattenPlan(base).map((t) => cellKey(t.condition)));
      expect(flattenPlan(alt).every((t) => t.condition.posture === 'sitting')).toBe(true);
    }
  });

  it('is deterministic, plain JSON and schema-valid, and pins one plan', () => {
    for (const seed of SEEDS) {
      const plan = street(seed);
      expect(street(seed)).toEqual(plan);
      expect(JSON.parse(JSON.stringify(plan))).toEqual(plan);
    }
    const plan = street(12345); // odd → left first; bit 1 = 0 → top_left first
    expect(plan.map((b) => `${b.tier}:${b.kind}:${b.hand}:${b.orientation}${b.optional ? '+' : ''}`)).toEqual([
      '0:choice:left:free',
      '1:main:left:upright_toward',
      '1:main:right:upright_toward',
      '2:main:right:top_left_toward',
      '2:main:left:top_left_toward',
      '2:main:left:top_right_toward',
      '2:main:right:top_right_toward',
      '3:odd:right:upright_toward',
      '3:odd:left:upright_toward',
      '4:main:left:upright_toward+',
      '4:main:right:upright_toward+',
      '4:odd:right:upright_toward+',
      '4:odd:left:upright_toward+',
    ]);
    expect(plan[7].trials.map((t) => cellKey(t.condition))).toEqual([
      'natural|natural|curved',
      'natural|center|standard',
      'natural|natural|pause',
      'natural|natural|tilted_touch',
    ]);
    expect(plan[11].trials.map((t) => t.condition.variant)).toEqual(['twist', 'lean_in']);
    expect(plan[1].trials.map((t) => `${t.condition.speed}/${t.condition.rep}`)).toEqual(['fast/0', 'natural/0', 'natural/1', 'slow/0']);
  });

  it('estimateMinutes counts only the non-optional blocks: 5 min (between 4 and 5.5) for 29 of the 39 trials', () => {
    for (const seed of SEEDS) {
      const plan = street(seed);
      const minutes = estimateMinutes(plan);
      expect(minutes).toBeGreaterThanOrEqual(4);
      expect(minutes).toBeLessThanOrEqual(5.5);
      expect(minutes).toBe(5);
      expect(flattenPlan(plan.filter((b) => !isOptionalBlock(b)))).toHaveLength(29);
      expect(flattenPlan(plan)).toHaveLength(39);
    }
  });
});

describe('buildPlan — seeding and counterbalancing', () => {
  it('first hand comes from the seed parity unless given', () => {
    for (const seed of SEEDS) {
      const expected: Hand = seed % 2 === 1 ? 'left' : 'right';
      expect(resolveFirstHand({ preset: 'quick', seed, posture: 'standing' })).toBe(expected);
      expect(buildPlan({ preset: 'quick', seed, posture: 'standing' })[0].hand).toBe(expected);
      expect(buildPlan({ preset: 'street', seed, posture: 'standing' })[0].hand).toBe(expected);
      for (const firstHand of ['left', 'right'] as Hand[]) {
        expect(resolveFirstHand({ preset: 'quick', seed, posture: 'standing', firstHand })).toBe(firstHand);
        expect(buildPlan({ preset: 'full', seed, posture: 'standing', firstHand })[0].hand).toBe(firstHand);
        expect(buildPlan({ preset: 'street', seed, posture: 'standing', firstHand })[1].hand).toBe(firstHand);
      }
    }
  });

  it('random session seeds balance the first hand across participants', () => {
    const rng = makeRng(1);
    let left = 0;
    for (let i = 0; i < 4000; i++) {
      const seed = Math.floor(rng() * 2 ** 32);
      if (resolveFirstHand({ preset: 'standard', seed, posture: 'standing' }) === 'left') left++;
    }
    expect(left / 4000).toBeGreaterThan(0.46);
    expect(left / 4000).toBeLessThan(0.54);
  });

  it('is deterministic: same options → deep-equal plan, and the plan is plain JSON', () => {
    for (const preset of FIXED_PRESETS) {
      for (const seed of SEEDS) {
        const opts: ProtocolOptions = { preset, seed, posture: 'sitting' };
        const plan = buildPlan(opts);
        expect(buildPlan({ ...opts })).toEqual(plan);
        expect(JSON.parse(JSON.stringify(plan))).toEqual(plan);
      }
    }
    const spec: CustomProtocolSpec = { ...DEFAULT_CUSTOM_SPEC, reps: 3, odd_variants: ['pause'], include_free: true };
    expect(buildPlan({ preset: 'custom', seed: 5, posture: 'standing', custom: spec })).toEqual(
      buildPlan({ preset: 'custom', seed: 5, posture: 'standing', custom: structuredClone(spec) }),
    );
  });

  it('different seeds give different orientation orders and trial orders', () => {
    const orientationOrders = new Set<string>();
    const firstCells = new Set<string>();
    const positions = new Map<OrientationId, Set<number>>();
    for (let seed = 0; seed < 200; seed++) {
      const plan = buildPlan({ preset: 'standard', seed, posture: 'standing' });
      const order = mainBlocks(plan)
        .filter((_, i) => i % 2 === 0)
        .map((b) => b.orientation);
      orientationOrders.add(order.join(','));
      order.forEach((o, i) => positions.set(o, (positions.get(o) ?? new Set()).add(i)));
      firstCells.add(cellKey(plan[0].trials[0].condition));
    }
    expect(orientationOrders.size).toBe(6); // all 3! orders of the non-upright orientations occur
    for (const o of CORE_ORIENTATIONS.slice(1)) expect(positions.get(o)).toEqual(new Set([1, 2, 3]));
    expect(firstCells.size).toBe(4); // any condition can open the session
    // same parity (same first hand) but different seeds still differ
    expect(buildPlan({ preset: 'full', seed: 2, posture: 'standing' })).not.toEqual(buildPlan({ preset: 'full', seed: 4, posture: 'standing' }));
  });

  it('the two hands of an orientation get independently shuffled trial orders', () => {
    let differ = 0;
    for (let seed = 0; seed < 50; seed++) {
      const [x, y] = buildPlan({ preset: 'full', seed, posture: 'standing' });
      if (x.trials.map((t) => cellKey(t.condition)).join() !== y.trials.map((t) => cellKey(t.condition)).join()) differ++;
    }
    expect(differ).toBeGreaterThan(40);
  });

  it('firstHand and posture do not change the random orders', () => {
    const base = buildPlan({ preset: 'full', seed: 77, posture: 'standing', firstHand: 'right' });
    const alt = buildPlan({ preset: 'full', seed: 77, posture: 'sitting', firstHand: 'left' });
    expect(alt.map((b) => [b.kind, b.orientation, other(b.hand)])).toEqual(base.map((b) => [b.kind, b.orientation, b.hand]));
    expect(flattenPlan(alt).map((t) => cellKey(t.condition))).toEqual(flattenPlan(base).map((t) => cellKey(t.condition)));
    expect(flattenPlan(alt).every((t) => t.condition.posture === 'sitting')).toBe(true);
  });

  it('rejects an unknown preset (inherited object keys included)', () => {
    expect(() => buildPlan({ preset: 'huge' as never, seed: 1, posture: 'standing' })).toThrow(/preset/);
    expect(() => buildPlan({ preset: 'toString' as never, seed: 1, posture: 'standing' })).toThrow(/preset/);
  });
});

describe('buildPlan — custom protocols', () => {
  const custom = (spec: Partial<CustomProtocolSpec>, seed = 11, posture: Posture = 'standing'): ProtocolOptions => ({
    preset: 'custom',
    seed,
    posture,
    custom: { ...DEFAULT_CUSTOM_SPEC, ...spec },
  });

  it('DEFAULT_CUSTOM_SPEC → 3 screen-toward orientations × 2 hands × 3 speeds; a missing spec means the default', () => {
    expect(DEFAULT_CUSTOM_SPEC.orientations).toEqual(['upright_toward', 'top_left_toward', 'top_right_toward']);
    const opts = custom({});
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan).toHaveLength(6);
    expect(flattenPlan(plan)).toHaveLength(18);
    expect(buildPlan({ preset: 'custom', seed: 11, posture: 'standing' })).toEqual(plan);
    expect(buildPlan({ preset: 'custom', seed: 11, posture: 'standing', custom: null })).toEqual(plan);
  });

  it('is ignored by the fixed presets', () => {
    const spec: CustomProtocolSpec = { ...DEFAULT_CUSTOM_SPEC, reps: 5, include_free: true };
    for (const preset of FIXED_PRESETS) {
      expect(buildPlan({ preset, seed: 3, posture: 'standing', custom: spec })).toEqual(buildPlan({ preset, seed: 3, posture: 'standing' }));
    }
  });

  it('screen-away grips are dropped like unknown values; a spec with nothing else falls back to the default grips', () => {
    const plan = buildPlan(custom({ orientations: ['upright_away', 'top_left_toward', 'upside_down_away'] }));
    expect(new Set(plan.map((b) => b.orientation))).toEqual(new Set(['top_left_toward']));
    const onlyAway = buildPlan(custom({ orientations: ['upright_away', 'top_right_away'] }));
    expect(new Set(onlyAway.map((b) => b.orientation))).toEqual(new Set(DEFAULT_CUSTOM_SPEC.orientations));
  });

  it('reps > 1 repeats every condition and `rep` counts the repeats in presentation order', () => {
    const opts = custom({
      orientations: ['upright_toward', 'top_left_toward'],
      speeds: ['natural', 'fast'],
      start_positions: ['shoulder', 'center'],
      reps: 3,
    });
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan).toHaveLength(4);
    for (const block of plan) {
      expect(block.trials).toHaveLength(12);
      const byCell = new Map<string, number[]>();
      for (const t of block.trials) byCell.set(cellKey(t.condition), [...(byCell.get(cellKey(t.condition)) ?? []), t.condition.rep]);
      expect(byCell.size).toBe(4);
      for (const reps of byCell.values()) expect(reps).toEqual([0, 1, 2]);
    }
    // shuffled: identical conditions are not simply grouped together in every block
    const grouped = plan.every((b) => b.trials.every((t, i) => i % 3 === 0 || cellKey(t.condition) === cellKey(b.trials[i - 1].condition)));
    expect(grouped).toBe(false);
  });

  it('odd_variants add one odd block per hand (upright_toward, natural speed and start) after the main blocks', () => {
    const opts = custom({ orientations: ['top_right_toward'], odd_variants: ['pause', 'chin', 'tilted_touch'] });
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan.map((b) => b.kind)).toEqual(['main', 'main', 'odd', 'odd']);
    for (const block of plan.slice(2)) {
      expect(block.orientation).toBe('upright_toward');
      expect(block.trials.map((t) => t.condition.variant).sort()).toEqual(['chin', 'pause', 'tilted_touch']);
      expect(block.trials.every((t) => t.condition.speed === 'natural' && t.condition.start_position === 'natural')).toBe(true);
    }
  });

  it('include_free adds free blocks last (A then B) with natural, fast and slow', () => {
    const opts = custom({ include_free: true, odd_variants: ['twist'] }, 8);
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan.map((b) => b.kind)).toEqual(['main', 'main', 'main', 'main', 'main', 'main', 'odd', 'odd', 'free', 'free']);
    expect(plan.slice(-2).map((b) => b.hand)).toEqual(['right', 'left']); // seed 8 → right first
    for (const block of plan.slice(-2)) {
      expect(block.orientation).toBe('free');
      expect(block.trials.map((t) => t.condition.speed).sort()).toEqual(['fast', 'natural', 'slow']);
    }
  });

  it('upright_toward leads only when it is selected; "free" may also be a main orientation', () => {
    const without = buildPlan(custom({ orientations: ['upside_down_toward', 'flat_up'] }, 21));
    expect(['upside_down_toward', 'flat_up']).toContain(without[0].orientation);
    for (let seed = 0; seed < 30; seed++) {
      const plan = buildPlan(custom({ orientations: ['flat_up', 'top_left_toward', 'upright_toward'] }, seed));
      expect(plan[0].orientation).toBe('upright_toward');
    }
    const free = buildPlan(custom({ orientations: ['free'] }));
    expect(free.map((b) => [b.kind, b.orientation])).toEqual([
      ['main', 'free'],
      ['main', 'free'],
    ]);
  });

  it('trial count = orientations × 2 × speeds × starts × reps + 2 × odd variants + 6 × include_free (random specs)', () => {
    const rng = makeRng(2026);
    const subset = <T>(all: readonly T[], min: number): T[] => {
      const picked = all.filter(() => rng() < 0.5);
      return picked.length >= min ? picked : shuffle(all, rng).slice(0, Math.max(min, 1));
    };
    for (let i = 0; i < 60; i++) {
      const spec: CustomProtocolSpec = {
        orientations: subset(COLLECTED_ORIENTATIONS, 1),
        speeds: subset(SPEEDS, 1),
        start_positions: subset(START_POSITIONS, 1),
        reps: 1 + Math.floor(rng() * MAX_CUSTOM_REPS),
        odd_variants: subset(VARIANTS, 0).slice(0, Math.floor(rng() * 4)),
        include_free: rng() < 0.5,
      };
      const opts: ProtocolOptions = { preset: 'custom', seed: Math.floor(rng() * 2 ** 32), posture: 'standing', custom: spec };
      const plan = buildPlan(opts);
      expectWellFormed(plan, opts);
      const expected =
        spec.orientations.length * 2 * spec.speeds.length * spec.start_positions.length * spec.reps +
        2 * spec.odd_variants.length +
        (spec.include_free ? 6 : 0);
      expect(flattenPlan(plan)).toHaveLength(expected);
      expect(estimateMinutes(plan) * 2).toBe(Math.round(estimateMinutes(plan) * 2));
    }
  });
});

describe('normalizeCustomSpec', () => {
  it('drops unknown values and duplicates, uses canonical order, clamps reps', () => {
    const messy = {
      orientations: ['top_left_toward', 'upright_toward', 'top_left_toward', 'sideways', 'top_left_away'],
      speeds: ['slow', 'natural', 'slow'],
      start_positions: ['center', 'natural'],
      reps: 2.7,
      odd_variants: ['jerky', 'standard', 'nope', 'curved', 'tilted_touch'],
      include_free: 1,
    } as unknown as CustomProtocolSpec;
    const before = JSON.stringify(messy);
    expect(normalizeCustomSpec(messy)).toEqual({
      orientations: ['upright_toward', 'top_left_toward'],
      speeds: ['natural', 'slow'],
      start_positions: ['natural', 'center'],
      reps: 2,
      odd_variants: ['standard', 'curved', 'jerky', 'tilted_touch'],
      include_free: false,
    });
    expect(JSON.stringify(messy)).toBe(before); // input untouched
  });

  it('falls back to the defaults for empty required lists and bad reps', () => {
    const empty: CustomProtocolSpec = { orientations: [], speeds: [], start_positions: [], reps: 0, odd_variants: [], include_free: true };
    expect(normalizeCustomSpec(empty)).toEqual({ ...DEFAULT_CUSTOM_SPEC, include_free: true });
    const reps = (r: unknown): number => normalizeCustomSpec({ ...DEFAULT_CUSTOM_SPEC, reps: r as number }).reps;
    expect([reps(-5), reps(0), reps(1), reps(10), reps(11), reps(1e9), reps(Number.NaN), reps('3'), reps(undefined)]).toEqual([
      1, 1, 1, 10, 10, 10, 1, 3, 1,
    ]);
    const fromNull = normalizeCustomSpec(null);
    expect(fromNull).toEqual(DEFAULT_CUSTOM_SPEC);
    expect(fromNull.orientations).not.toBe(DEFAULT_CUSTOM_SPEC.orientations); // a copy, safe to edit
    expect(normalizeCustomSpec(undefined)).toEqual(DEFAULT_CUSTOM_SPEC);
  });

  it('accepts exactly the collected orientations (every id except the screen-away grips)', () => {
    expect(COLLECTED_ORIENTATIONS).toEqual(['upright_toward', 'top_left_toward', 'top_right_toward', 'upside_down_toward', 'flat_up', 'free']);
    expect(COLLECTED_ORIENTATIONS.filter(isAway)).toEqual([]);
    expect(ALL_ORIENTATIONS.filter((o) => !COLLECTED_ORIENTATIONS.includes(o)).every(isAway)).toBe(true);
    expect(normalizeCustomSpec({ ...DEFAULT_CUSTOM_SPEC, orientations: [...ALL_ORIENTATIONS] }).orientations).toEqual(COLLECTED_ORIENTATIONS);
  });

  it('always returns a schema-valid spec, is idempotent, and buildPlan only depends on the chosen sets', () => {
    const rng = makeRng(99);
    const junk = ['bogus', '', 'LEFT', 'upright', 'toString'];
    const pick = <T>(xs: readonly T[]): T[] => xs.filter(() => rng() < 0.4);
    for (let i = 0; i < 80; i++) {
      const raw = {
        orientations: shuffle([...pick(ALL_ORIENTATIONS), ...pick(junk)], rng),
        speeds: shuffle([...pick(SPEEDS), ...pick(SPEEDS), ...pick(junk)], rng),
        start_positions: shuffle([...pick(START_POSITIONS), ...pick(junk)], rng),
        reps: Math.floor(rng() * 30) - 10,
        odd_variants: shuffle([...pick(VARIANTS), ...pick(junk)], rng),
        include_free: rng() < 0.5,
      } as unknown as CustomProtocolSpec;
      const norm = normalizeCustomSpec(raw);
      expect(schemaErrors('/$defs/CustomProtocolSpec', norm)).toEqual([]);
      expect(norm.orientations.filter(isAway)).toEqual([]);
      expect(normalizeCustomSpec(norm)).toEqual(norm);
      const seed = i * 7919;
      const plan = buildPlan({ preset: 'custom', seed, posture: 'standing', custom: raw });
      expect(plan).toEqual(buildPlan({ preset: 'custom', seed, posture: 'standing', custom: norm }));
      const permuted: CustomProtocolSpec = {
        ...norm,
        orientations: shuffle(norm.orientations, rng),
        speeds: shuffle(norm.speeds, rng),
        start_positions: shuffle(norm.start_positions, rng),
        odd_variants: shuffle(norm.odd_variants, rng),
      };
      expect(buildPlan({ preset: 'custom', seed, posture: 'standing', custom: permuted })).toEqual(plan);
    }
  });

  it('the condition enums match the schema', () => {
    expect([...SPEEDS].sort()).toEqual([...recordsSchema.$defs.Speed.enum].sort());
    expect([...START_POSITIONS].sort()).toEqual([...recordsSchema.$defs.StartPosition.enum].sort());
    expect([...VARIANTS].sort()).toEqual([...recordsSchema.$defs.Variant.enum].sort());
    expect(MAX_CUSTOM_REPS).toBe(recordsSchema.$defs.CustomProtocolSpec.properties.reps.maximum);
    expect([...FIXED_PRESETS, 'custom'].sort()).toEqual([...recordsSchema.$defs.ProtocolPreset.enum].sort());
  });
});

describe('flattenPlan', () => {
  it('concatenates the blocks in order; index i holds trial_index i', () => {
    const plan = buildPlan({ preset: 'full', seed: 5, posture: 'standing' });
    const flat = flattenPlan(plan);
    expect(flat.map((t) => t.trial_index)).toEqual(Array.from({ length: 86 }, (_, i) => i));
    expect(flat).toEqual(plan.flatMap((b) => b.trials));
    expect(flat[0]).toBe(plan[0].trials[0]); // same objects, no copies
    expect(flattenPlan([])).toEqual([]);
  });
});

describe('estimateMinutes', () => {
  type Cond = Partial<Pick<TrialCondition, 'speed' | 'start_position' | 'variant'>>;
  /** A block of trials with the given conditions (natural / natural / standard by default). */
  const blockOf = (conds: Cond[], index = 0, extra: Partial<PlannedBlock> = {}): PlannedBlock => ({
    block_index: index,
    kind: 'main',
    hand: 'right',
    orientation: 'upright_toward',
    trials: conds.map((c, i) => ({
      trial_index: i,
      block_index: index,
      trial_in_block: i,
      condition: {
        hand: 'right' as Hand,
        orientation: 'upright_toward' as OrientationId,
        speed: 'natural' as Speed,
        start_position: 'natural' as StartPosition,
        variant: 'standard' as Variant,
        posture: 'standing' as Posture,
        rep: 0,
        ...c,
      },
    })),
    ...extra,
  });
  const block = (n: number, index = 0, extra: Partial<PlannedBlock> = {}): PlannedBlock =>
    blockOf(Array.from({ length: n }, () => ({})), index, extra);
  const times = (n: number, c: Cond): Cond[] => Array.from({ length: n }, () => c);

  it('street ≈ 5, quick 3.5, standard 8, full 15 minutes', () => {
    const minutes = (preset: (typeof FIXED_PRESETS)[number]): number => estimateMinutes(buildPlan({ preset, seed: 1, posture: 'standing' }));
    expect(minutes('street')).toBeGreaterThanOrEqual(4);
    expect(minutes('street')).toBeLessThanOrEqual(5.5);
    for (const preset of FIXED_PRESETS) expect(minutes(preset)).toBe(EXPECTED[preset].minutes);
    expect(estimateMinutes(buildPlan({ preset: 'custom', seed: 1, posture: 'standing' }))).toBe(3.5);
  });

  it('follows 40 s setup + 6.5 s per natural trial + 6 s per block change, rounded to 0.5 min', () => {
    expect(estimateMinutes([])).toBe(0.5); // 40 s
    expect(estimateMinutes([block(1)])).toBe(1); // 46.5 s = 0.78 min
    expect(estimateMinutes([block(3), block(3, 1)])).toBe(1.5); // 40 + 39 + 6 = 85 s = 1.42 min
    expect(estimateMinutes([block(10)])).toBe(2); // 105 s = 1.75 min, halves round up
    expect(estimateMinutes([block(3), block(0, 1), block(3, 2)])).toBe(1.5); // empty blocks are not shown
    expect(estimateMinutes(Array.from({ length: 20 }, (_, i) => block(1, i)))).toBe(4.5); // 40 + 130 + 114 = 284 s = 4.73 min
  });

  it('weights the trials by speed and adds 2 s once for a non-default start position or variant', () => {
    expect(estimateMinutes([block(60)])).toBe(7); // 40 + 390 = 430 s = 7.17 min
    expect(estimateMinutes([blockOf(times(60, { speed: 'fast' }))])).toBe(6); // 40 + 330 = 370 s = 6.17 min
    expect(estimateMinutes([blockOf(times(60, { speed: 'slow' }))])).toBe(10.5); // 40 + 600 = 640 s = 10.67 min
    expect(estimateMinutes([blockOf(times(60, { variant: 'curved' }))])).toBe(9); // 40 + 510 = 550 s = 9.17 min
    expect(estimateMinutes([blockOf(times(60, { start_position: 'center' }))])).toBe(9);
    expect(estimateMinutes([blockOf(times(60, { start_position: 'shoulder', variant: 'pause' }))])).toBe(9); // +2 s once
    expect(estimateMinutes([blockOf(times(60, { speed: 'slow', variant: 'tilted_touch' }))])).toBe(12.5); // 40 + 720 = 760 s = 12.67 min
  });

  it('ignores optional (bonus) blocks', () => {
    expect(estimateMinutes([block(10), block(10, 1, { optional: true })])).toBe(2);
    expect(estimateMinutes([block(10), block(10, 1, { optional: false })])).toBe(3); // 40 + 130 + 6 = 176 s = 2.93 min
    expect(isOptionalBlock(block(1, 0, { optional: true }))).toBe(true);
    expect(isOptionalBlock(block(1))).toBe(false);
  });

  it('produces a session `protocol` object that validates against the SessionRecord schema', () => {
    for (const preset of [...FIXED_PRESETS, 'custom'] as const) {
      const seed = 4242;
      const custom = preset === 'custom' ? normalizeCustomSpec({ ...DEFAULT_CUSTOM_SPEC, odd_variants: ['lean_in', 'tilted_touch'] }) : null;
      const opts: ProtocolOptions = { preset, seed, posture: 'sitting', custom };
      const plan = buildPlan(opts);
      const protocol = {
        preset,
        protocol_version: PROTOCOL_VERSION,
        seed,
        posture: opts.posture,
        first_hand: resolveFirstHand(opts),
        n_planned_trials: flattenPlan(plan).length,
        estimated_minutes: estimateMinutes(plan),
        custom,
        plan,
      };
      expect(schemaErrors('/$defs/SessionRecord/properties/protocol', protocol)).toEqual([]);
    }
  });
});

describe('resolveParams', () => {
  const D = DEFAULT_SEGMENTATION_PARAMS;
  const cond = (speed: Speed, variant: Variant = 'standard'): TrialCondition => ({
    hand: 'left',
    orientation: 'upright_toward',
    speed,
    start_position: 'natural',
    variant,
    posture: 'standing',
    rep: 0,
  });
  const SLOW = {
    onset_gyro_min_dps: 25,
    onset_acc_min_ms2: 0.55,
    end_gyro_dps: 10,
    end_acc_ms2: 0.25,
    end_sustain_ms: 450,
    max_movement_ms: 12000,
  };

  it('natural + standard returns an equal copy of the base', () => {
    const p = resolveParams(D, cond('natural'));
    expect(p).toEqual(D);
    expect(p).not.toBe(D);
  });

  it('applies the documented speed and variant overrides', () => {
    expect(resolveParams(D, cond('slow'))).toEqual({ ...D, ...SLOW });
    expect(resolveParams(D, cond('fast'))).toEqual({ ...D, onset_sustain_ms: 40 });
    expect(resolveParams(D, cond('natural', 'pause'))).toEqual({ ...D, end_sustain_ms: 1800, max_movement_ms: 12000 });
    expect(resolveParams(D, cond('natural', 'jerky'))).toEqual({ ...D, end_sustain_ms: 600 });
    for (const v of ['curved', 'twist', 'elbow_high', 'bent_start', 'lean_in', 'chin', 'forehead', 'tilted_touch'] as Variant[]) {
      expect(resolveParams(D, cond('natural', v))).toEqual(D);
      expect(resolveParams(D, cond('slow', v))).toEqual({ ...D, ...SLOW });
    }
  });

  it('keeps the slow onset floors above the still thresholds (a slow sway is not a false start)', () => {
    const slow = resolveParams(D, cond('slow'));
    expect(slow.onset_gyro_min_dps).toBeGreaterThan(D.still_gyro_dps);
    expect(slow.onset_acc_min_ms2).toBeGreaterThan(D.still_acc_ms2);
    expect(slow.return_cap_ms).toBe(D.return_cap_ms); // parameters no override names are copied
  });

  it('composes speed and variant overrides', () => {
    // slow + pause → slow thresholds, pause's end_sustain_ms wins, max of max_movement_ms
    expect(resolveParams(D, cond('slow', 'pause'))).toEqual({ ...D, ...SLOW, end_sustain_ms: 1800, max_movement_ms: 12000 });
    expect(resolveParams(D, cond('slow', 'jerky'))).toEqual({ ...D, ...SLOW, end_sustain_ms: 600 });
    expect(resolveParams(D, cond('fast', 'pause'))).toEqual({ ...D, onset_sustain_ms: 40, end_sustain_ms: 1800, max_movement_ms: 12000 });
    expect(resolveParams(D, cond('fast', 'jerky'))).toEqual({ ...D, onset_sustain_ms: 40, end_sustain_ms: 600 });
  });

  it('never mutates the base (frozen base, every condition)', () => {
    const base = Object.freeze({ ...D });
    const before = JSON.stringify(base);
    for (const speed of SPEEDS) for (const v of VARIANTS) resolveParams(base, cond(speed, v)); // would throw on write
    expect(JSON.stringify(base)).toBe(before);
    expect(JSON.stringify(D)).toBe(before);
  });

  it('only moves a tuned base in the override direction (thresholds down, waits up)', () => {
    const tuned: SegmentationParams = { ...D, max_movement_ms: 20000, onset_gyro_min_dps: 10, end_sustain_ms: 2500, onset_sustain_ms: 20 };
    const slowPause = resolveParams(tuned, cond('slow', 'pause'));
    expect(slowPause.max_movement_ms).toBe(20000); // a slow trial never gets LESS time than the base
    expect(slowPause.onset_gyro_min_dps).toBe(10); // ...nor a stricter onset floor
    expect(slowPause.end_sustain_ms).toBe(2500);
    expect(resolveParams(tuned, cond('fast')).onset_sustain_ms).toBe(20);
    const direction: Partial<Record<keyof SegmentationParams, 'down' | 'up'>> = {
      onset_gyro_min_dps: 'down',
      onset_acc_min_ms2: 'down',
      onset_sustain_ms: 'down',
      end_gyro_dps: 'down',
      end_acc_ms2: 'down',
      end_sustain_ms: 'up',
      max_movement_ms: 'up',
    };
    const rng = makeRng(8);
    const violations: string[] = [];
    for (let i = 0; i < 200; i++) {
      const base = Object.fromEntries(Object.entries(D).map(([k, v]) => [k, v * (0.25 + rng() * 3.75)])) as unknown as SegmentationParams;
      for (const speed of SPEEDS) {
        for (const v of VARIANTS) {
          const p = resolveParams(base, cond(speed, v));
          for (const k of Object.keys(base) as (keyof SegmentationParams)[]) {
            const pk = p[k] as number;
            const bk = base[k] as number;
            const ok = direction[k] === 'down' ? pk <= bk : direction[k] === 'up' ? pk >= bk : pk === bk;
            if (!ok) violations.push(`${speed}/${v} ${k}: base ${bk} → ${pk}`);
          }
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it('every planned condition resolves to schema-valid parameters', () => {
    for (const preset of FIXED_PRESETS) {
      for (const t of flattenPlan(buildPlan({ preset, seed: 17, posture: 'standing' }))) {
        expect(schemaErrors('/$defs/SegmentationParams', resolveParams(D, t.condition))).toEqual([]);
      }
    }
  });
});
