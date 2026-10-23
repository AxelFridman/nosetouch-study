import { Ajv2020 } from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';
import {
  ALL_ORIENTATIONS,
  CORE_ORIENTATIONS,
  DEFAULT_SEGMENTATION_PARAMS,
  ODD_VARIANTS,
  PROTOCOL_VERSION,
} from '../src/core/constants';
import {
  DEFAULT_CUSTOM_SPEC,
  MAX_CUSTOM_REPS,
  SPEEDS,
  START_POSITIONS,
  VARIANTS,
  buildPlan,
  estimateMinutes,
  flattenPlan,
  normalizeCustomSpec,
  presetTrialCount,
  resolveFirstHand,
  resolveParams,
} from '../src/core/protocol';
import type {
  CustomProtocolSpec,
  Hand,
  OrientationId,
  PlannedBlock,
  Posture,
  ProtocolOptions,
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

const PRESETS = ['quick', 'standard', 'full'] as const;
const SEEDS = [0, 1, 2, 3, 7, 42, 12345, 123456789, 2 ** 31, 2 ** 32 - 1];
const POSTURES: Posture[] = ['standing', 'sitting'];
const EXPECTED = {
  quick: { trials: 24, blocks: 8, minutes: 7.5 },
  standard: { trials: 56, blocks: 12, minutes: 14 },
  full: { trials: 132, blocks: 22, minutes: 29.5 },
} as const;
const NON_FREE = ALL_ORIENTATIONS.filter((o) => o !== 'free');
const STANDARD_ODD: Variant[] = ['curved', 'pause', 'twist', 'bent_start', 'chin', 'lean_in', 'elbow_high', 'jerky'];

const other = (h: Hand): Hand => (h === 'left' ? 'right' : 'left');
const key = (...parts: string[]): string => parts.join('|');
const cellKey = (c: TrialCondition): string => key(c.speed, c.start_position, c.variant);
const mainBlocks = (plan: PlannedBlock[]): PlannedBlock[] => plan.filter((b) => b.kind === 'main');

/** Multiset of keys → counts. */
function countBy<T>(items: T[], f: (x: T) => string): Map<string, number> {
  const m = new Map<string, number>();
  for (const x of items) m.set(f(x), (m.get(f(x)) ?? 0) + 1);
  return m;
}

/** Every structural rule of the generator (see protocol.ts ORDERING RULES). */
function expectWellFormed(plan: PlannedBlock[], opts: ProtocolOptions): void {
  const a = resolveFirstHand(opts);
  const b = other(a);
  const phase = { main: 0, odd: 1, free: 2 } as const;
  let lastPhase = 0;
  let next = 0;
  const conditions = new Set<TrialCondition>();
  plan.forEach((block, bi) => {
    expectValid('PlannedBlock', block);
    expect(block.block_index).toBe(bi);
    expect(block.trials.length).toBeGreaterThan(0);
    expect(phase[block.kind]).toBeGreaterThanOrEqual(lastPhase); // main → odd → free
    lastPhase = phase[block.kind];
    if (block.kind === 'odd') expect(block.orientation).toBe('upright_toward');
    if (block.kind === 'free') expect(block.orientation).toBe('free');
    const reps = new Map<string, number>();
    block.trials.forEach((t, ti) => {
      expect(t.trial_index).toBe(next++);
      expect(t.block_index).toBe(bi);
      expect(t.trial_in_block).toBe(ti);
      const c = t.condition;
      expect(c.hand).toBe(block.hand);
      expect(c.orientation).toBe(block.orientation);
      expect(c.posture).toBe(opts.posture);
      // rep = number of earlier identical conditions in this block (presentation order)
      expect(c.rep).toBe(reps.get(cellKey(c)) ?? 0);
      reps.set(cellKey(c), (reps.get(cellKey(c)) ?? 0) + 1);
      if (block.kind !== 'odd') expect(c.variant).toBe('standard');
      if (block.kind !== 'main') expect(c.start_position).toBe('natural');
      if (block.kind === 'odd') expect(c.speed).toBe('natural');
      expect(conditions.has(c)).toBe(false); // no shared condition objects
      conditions.add(c);
    });
  });
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
  it.each(PRESETS)('%s has the documented trial and block counts for every seed', (preset) => {
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

  it('quick: 4 orientations × 2 hands × {natural, fast, slow}, start natural, standard variant', () => {
    const plan = buildPlan({ preset: 'quick', seed: 9, posture: 'standing' });
    expect(plan.every((b) => b.kind === 'main')).toBe(true);
    const counts = countBy(flattenPlan(plan), (t) => key(t.condition.hand, t.condition.orientation, cellKey(t.condition)));
    expect(counts.size).toBe(24);
    for (const hand of ['left', 'right']) {
      for (const o of ['upright_toward', 'upright_away', 'top_left_toward', 'top_right_toward']) {
        for (const speed of ['natural', 'fast', 'slow']) expect(counts.get(key(hand, o, speed, 'natural', 'standard'))).toBe(1);
      }
    }
  });

  it('standard: every (hand, orientation, speed, start_position) main condition exactly once; odd blocks hold the 8 listed variants per hand', () => {
    for (const seed of SEEDS) {
      const plan = buildPlan({ preset: 'standard', seed, posture: 'sitting' });
      const mainTrials = flattenPlan(mainBlocks(plan));
      const counts = countBy(mainTrials, (t) =>
        key(t.condition.hand, t.condition.orientation, t.condition.speed, t.condition.start_position),
      );
      expect(mainTrials).toHaveLength(40);
      expect(counts.size).toBe(40);
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

  it('full: 9 orientations × 2 hands × 6 cells once each, all 9 odd variants per hand, free blocks last', () => {
    const plan = buildPlan({ preset: 'full', seed: 314, posture: 'standing' });
    const main = flattenPlan(mainBlocks(plan));
    const counts = countBy(main, (t) => key(t.condition.hand, t.condition.orientation, cellKey(t.condition)));
    expect(main).toHaveLength(108);
    expect(counts.size).toBe(108);
    for (const hand of ['left', 'right']) {
      for (const o of NON_FREE) {
        for (const sp of ['natural', 'fast', 'slow']) {
          for (const st of ['shoulder', 'center']) expect(counts.get(key(hand, o, sp, st, 'standard'))).toBe(1);
        }
      }
    }
    const odd = plan.filter((b) => b.kind === 'odd');
    expect(odd).toHaveLength(2);
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
    expect(plan.map((b) => b.hand)).toEqual([R, L, L, R, R, L, L, R, R, L, R, L]);
    const swapped = buildPlan({ preset: 'standard', seed: 4, posture: 'standing', firstHand: 'left' });
    expect(swapped.map((b) => b.hand)).toEqual([L, R, R, L, L, R, R, L, L, R, L, R]);
  });

  it('pins the generated plan for one seed (a change here needs a PROTOCOL_VERSION bump)', () => {
    expect(PROTOCOL_VERSION).toBe('1.0.0');
    const plan = buildPlan({ preset: 'standard', seed: 12345, posture: 'standing' });
    expect(plan.map((b) => `${b.kind}:${b.hand}:${b.orientation}`)).toEqual([
      'main:left:upright_toward',
      'main:right:upright_toward',
      'main:right:upright_away',
      'main:left:upright_away',
      'main:left:top_left_toward',
      'main:right:top_left_toward',
      'main:right:upside_down_toward',
      'main:left:upside_down_toward',
      'main:left:top_right_toward',
      'main:right:top_right_toward',
      'odd:left:upright_toward',
      'odd:right:upright_toward',
    ]);
    expect(plan[0].trials.map((t) => `${t.condition.speed}/${t.condition.start_position}`)).toEqual([
      'natural/center',
      'slow/natural',
      'fast/natural',
      'natural/shoulder',
    ]);
    expect(plan[10].trials.map((t) => t.condition.variant)).toEqual([
      'elbow_high',
      'chin',
      'curved',
      'bent_start',
      'lean_in',
      'twist',
      'jerky',
      'pause',
    ]);
  });
});

describe('buildPlan — seeding and counterbalancing', () => {
  it('first hand comes from the seed parity unless given', () => {
    for (const seed of SEEDS) {
      const expected: Hand = seed % 2 === 1 ? 'left' : 'right';
      expect(resolveFirstHand({ preset: 'quick', seed, posture: 'standing' })).toBe(expected);
      expect(buildPlan({ preset: 'quick', seed, posture: 'standing' })[0].hand).toBe(expected);
      for (const firstHand of ['left', 'right'] as Hand[]) {
        expect(resolveFirstHand({ preset: 'quick', seed, posture: 'standing', firstHand })).toBe(firstHand);
        expect(buildPlan({ preset: 'full', seed, posture: 'standing', firstHand })[0].hand).toBe(firstHand);
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
    for (const preset of PRESETS) {
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
    expect(orientationOrders.size).toBe(24); // all 4! orders of the non-upright orientations occur
    for (const o of CORE_ORIENTATIONS.slice(1)) expect(positions.get(o)).toEqual(new Set([1, 2, 3, 4]));
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

  it('rejects an unknown preset', () => {
    expect(() => buildPlan({ preset: 'huge' as never, seed: 1, posture: 'standing' })).toThrow(/preset/);
  });
});

describe('buildPlan — custom protocols', () => {
  const custom = (spec: Partial<CustomProtocolSpec>, seed = 11, posture: Posture = 'standing'): ProtocolOptions => ({
    preset: 'custom',
    seed,
    posture,
    custom: { ...DEFAULT_CUSTOM_SPEC, ...spec },
  });

  it('DEFAULT_CUSTOM_SPEC → 2 orientations × 2 hands × 3 speeds; a missing spec means the default', () => {
    const opts = custom({});
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan).toHaveLength(4);
    expect(flattenPlan(plan)).toHaveLength(12);
    expect(buildPlan({ preset: 'custom', seed: 11, posture: 'standing' })).toEqual(plan);
    expect(buildPlan({ preset: 'custom', seed: 11, posture: 'standing', custom: null })).toEqual(plan);
  });

  it('is ignored by the fixed presets', () => {
    const spec: CustomProtocolSpec = { ...DEFAULT_CUSTOM_SPEC, reps: 5, include_free: true };
    expect(buildPlan({ preset: 'standard', seed: 3, posture: 'standing', custom: spec })).toEqual(
      buildPlan({ preset: 'standard', seed: 3, posture: 'standing' }),
    );
  });

  it('reps > 1 repeats every condition and `rep` counts the repeats in presentation order', () => {
    const opts = custom({
      orientations: ['upright_toward', 'top_left_away'],
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
    const opts = custom({ orientations: ['top_right_away'], odd_variants: ['pause', 'chin', 'forehead'] });
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan.map((b) => b.kind)).toEqual(['main', 'main', 'odd', 'odd']);
    for (const block of plan.slice(2)) {
      expect(block.orientation).toBe('upright_toward');
      expect(block.trials.map((t) => t.condition.variant).sort()).toEqual(['chin', 'forehead', 'pause']);
      expect(block.trials.every((t) => t.condition.speed === 'natural' && t.condition.start_position === 'natural')).toBe(true);
    }
  });

  it('include_free adds free blocks last (A then B) with natural, fast and slow', () => {
    const opts = custom({ include_free: true, odd_variants: ['twist'] }, 8);
    const plan = buildPlan(opts);
    expectWellFormed(plan, opts);
    expect(plan.map((b) => b.kind)).toEqual(['main', 'main', 'main', 'main', 'odd', 'odd', 'free', 'free']);
    expect(plan.slice(-2).map((b) => b.hand)).toEqual(['right', 'left']); // seed 8 → right first
    for (const block of plan.slice(-2)) {
      expect(block.orientation).toBe('free');
      expect(block.trials.map((t) => t.condition.speed).sort()).toEqual(['fast', 'natural', 'slow']);
    }
  });

  it('upright_toward leads only when it is selected; "free" may also be a main orientation', () => {
    const without = buildPlan(custom({ orientations: ['upside_down_away', 'flat_up'] }, 21));
    expect(['upside_down_away', 'flat_up']).toContain(without[0].orientation);
    for (let seed = 0; seed < 30; seed++) {
      const plan = buildPlan(custom({ orientations: ['flat_up', 'upright_away', 'upright_toward'] }, seed));
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
        orientations: subset(ALL_ORIENTATIONS, 1),
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
      orientations: ['top_left_toward', 'upright_toward', 'top_left_toward', 'sideways'],
      speeds: ['slow', 'natural', 'slow'],
      start_positions: ['center', 'natural'],
      reps: 2.7,
      odd_variants: ['jerky', 'standard', 'nope', 'curved'],
      include_free: 1,
    } as unknown as CustomProtocolSpec;
    const before = JSON.stringify(messy);
    expect(normalizeCustomSpec(messy)).toEqual({
      orientations: ['upright_toward', 'top_left_toward'],
      speeds: ['natural', 'slow'],
      start_positions: ['natural', 'center'],
      reps: 2,
      odd_variants: ['standard', 'curved', 'jerky'],
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
  });
});

describe('flattenPlan', () => {
  it('concatenates the blocks in order; index i holds trial_index i', () => {
    const plan = buildPlan({ preset: 'full', seed: 5, posture: 'standing' });
    const flat = flattenPlan(plan);
    expect(flat.map((t) => t.trial_index)).toEqual(Array.from({ length: 132 }, (_, i) => i));
    expect(flat).toEqual(plan.flatMap((b) => b.trials));
    expect(flat[0]).toBe(plan[0].trials[0]); // same objects, no copies
    expect(flattenPlan([])).toEqual([]);
  });
});

describe('estimateMinutes', () => {
  const block = (n: number, index = 0): PlannedBlock => ({
    block_index: index,
    kind: 'main',
    hand: 'right',
    orientation: 'upright_toward',
    trials: Array.from({ length: n }, (_, i) => ({
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
      },
    })),
  });

  it('is roughly quick ≈ 5–8, standard 12–15, full ≈ 30 minutes (exact: 7.5 / 14 / 29.5)', () => {
    const minutes = (preset: (typeof PRESETS)[number]): number => estimateMinutes(buildPlan({ preset, seed: 1, posture: 'standing' }));
    expect(minutes('quick')).toBeGreaterThanOrEqual(4.5);
    expect(minutes('quick')).toBeLessThanOrEqual(8);
    expect(minutes('standard')).toBeGreaterThanOrEqual(12);
    expect(minutes('standard')).toBeLessThanOrEqual(15);
    expect(minutes('full')).toBeGreaterThanOrEqual(28);
    expect(minutes('full')).toBeLessThanOrEqual(32);
    for (const preset of PRESETS) expect(minutes(preset)).toBe(EXPECTED[preset].minutes);
  });

  it('follows 9 s/trial + 25 s/block change + 60 s setup, rounded to 0.5 min', () => {
    expect(estimateMinutes([])).toBe(1); // setup only
    expect(estimateMinutes([block(1)])).toBe(1); // 69 s = 1.15 min
    expect(estimateMinutes([block(3), block(3, 1)])).toBe(2.5); // 60 + 54 + 25 = 139 s = 2.32 min
    expect(estimateMinutes([block(10)])).toBe(2.5); // 150 s
    expect(estimateMinutes([block(3), block(0, 1), block(3, 2)])).toBe(2.5); // empty blocks are not shown
  });

  it('produces a session `protocol` object that validates against the SessionRecord schema', () => {
    for (const preset of [...PRESETS, 'custom'] as const) {
      const seed = 4242;
      const custom = preset === 'custom' ? normalizeCustomSpec({ ...DEFAULT_CUSTOM_SPEC, odd_variants: ['lean_in'] }) : null;
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
    onset_gyro_min_dps: 15,
    onset_acc_min_ms2: 0.3,
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
    for (const v of ['curved', 'twist', 'elbow_high', 'bent_start', 'lean_in', 'chin', 'forehead'] as Variant[]) {
      expect(resolveParams(D, cond('natural', v))).toEqual(D);
      expect(resolveParams(D, cond('slow', v))).toEqual({ ...D, ...SLOW });
    }
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
            const ok = direction[k] === 'down' ? p[k] <= base[k] : direction[k] === 'up' ? p[k] >= base[k] : p[k] === base[k];
            if (!ok) violations.push(`${speed}/${v} ${k}: base ${base[k]} → ${p[k]}`);
          }
        }
      }
    }
    expect(violations).toEqual([]);
  });

  it('every planned condition resolves to schema-valid parameters', () => {
    for (const preset of PRESETS) {
      for (const t of flattenPlan(buildPlan({ preset, seed: 17, posture: 'standing' }))) {
        expect(schemaErrors('/$defs/SegmentationParams', resolveParams(D, t.condition))).toEqual([]);
      }
    }
  });
});
