import { afterEach, describe, expect, it, vi } from 'vitest';
import { ROUNDING } from '../src/core/constants';
import type { EhiAnswer, EhiShortForm } from '../src/core/types';
import {
  EHI_ANSWER_SCORES,
  EHI_ITEMS,
  PARTICIPANT_ID_ALPHABET,
  ehiClassFromLq,
  isParticipantId,
  isoTime,
  makeRng,
  mixSeed,
  newId,
  newParticipantId,
  randomSeed,
  roundOrNull,
  scoreEhi,
  shuffle,
} from '../src/core/util';
import { recordsSchema } from './helpers/schema';

const UINT32 = 2 ** 32;
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const ID_RE = new RegExp(recordsSchema.$defs.Id.pattern);
const PID_RE = new RegExp(recordsSchema.$defs.ParticipantId.pattern);
const realCrypto = globalThis.crypto;

/** Textbook mulberry32 (Tommy Ettinger), used as the reference implementation. */
function mulberry32Reference(seed: number): () => number {
  let a = seed;
  return () => {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Exact n-th (1-based) mulberry32 output, computed from the closed-form state with BigInt. */
function mulberry32Nth(seed: number, n: number): number {
  const state = Number((BigInt(seed >>> 0) + BigInt(n) * 0x6d2b79f5n) % 2n ** 32n) | 0;
  let t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function draws(rng: () => number, n: number): number[] {
  return Array.from({ length: n }, () => rng());
}

/** Pearson chi-square statistic of observed counts against a uniform expectation. */
function chiSquare(counts: number[]): number {
  const total = counts.reduce((a, b) => a + b, 0);
  const expected = total / counts.length;
  return counts.reduce((s, c) => s + (c - expected) ** 2 / expected, 0);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('makeRng (mulberry32)', () => {
  it('is deterministic for a seed and matches the reference implementation', () => {
    for (const seed of [0, 1, 2, 42, 123456789, 2 ** 31, UINT32 - 1]) {
      const ref = mulberry32Reference(seed);
      expect(draws(makeRng(seed), 1000)).toEqual(draws(ref, 1000));
      expect(draws(makeRng(seed), 50)).toEqual(draws(makeRng(seed), 50));
    }
  });

  it('pins known output values (regression guard)', () => {
    const r = makeRng(0);
    expect(r()).toBe(0.26642920868471265);
    expect(r()).toBe(0.0003297457005828619);
    expect(r()).toBe(0.2232720274478197);
    const r42 = makeRng(42);
    expect([r42(), r42(), r42()]).toEqual([0.6011037519201636, 0.44829055899754167, 0.8524657934904099]);
  });

  it('different seeds give different streams', () => {
    const firsts = new Set(Array.from({ length: 1000 }, (_, s) => makeRng(s)()));
    expect(firsts.size).toBe(1000);
    expect(draws(makeRng(1), 20)).not.toEqual(draws(makeRng(2), 20));
  });

  it('takes seeds modulo 2^32 (negative, oversized and NaN seeds)', () => {
    expect(draws(makeRng(-1), 10)).toEqual(draws(makeRng(UINT32 - 1), 10));
    expect(draws(makeRng(UINT32 + 5), 10)).toEqual(draws(makeRng(5), 10));
    expect(draws(makeRng(Number.NaN), 10)).toEqual(draws(makeRng(0), 10));
  });

  it('returns floats in [0, 1) with a roughly uniform distribution', () => {
    const xs = draws(makeRng(2024), 200_000);
    const bins = new Array(20).fill(0);
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    for (const x of xs) {
      bins[Math.floor(x * 20)]++;
      sum += x;
      min = Math.min(min, x);
      max = Math.max(max, x);
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThan(1);
    expect(sum / xs.length).toBeGreaterThan(0.495);
    expect(sum / xs.length).toBeLessThan(0.505);
    // chi-square, 19 degrees of freedom: 43.8 is the p = 0.001 critical value
    expect(chiSquare(bins)).toBeLessThan(43.8);
  });

  it('stays exact past the point where the textbook float state loses precision (~4.9M calls)', () => {
    const seed = 987654321;
    const rng = makeRng(seed);
    const n = 5_000_000;
    let last = 0;
    for (let i = 0; i < n; i++) last = rng();
    expect(last).toBe(mulberry32Nth(seed, n));
    expect(rng()).toBe(mulberry32Nth(seed, n + 1));
  });
});

describe('mixSeed', () => {
  it('is a deterministic uint32 that depends on the salt', () => {
    for (const seed of [0, 1, 7, 2 ** 31, UINT32 - 1]) {
      const v = mixSeed(seed, 11);
      expect(Number.isInteger(v) && v >= 0 && v < UINT32).toBe(true);
      expect(mixSeed(seed, 11)).toBe(v);
      expect(mixSeed(seed, 12)).not.toBe(v);
    }
  });

  it('never collides for a fixed salt and scrambles consecutive seeds', () => {
    const out = Array.from({ length: 20_000 }, (_, s) => mixSeed(s, 0x504c414e));
    expect(new Set(out).size).toBe(out.length);
    const odd = out.filter((v) => v & 1).length / out.length;
    const high = out.filter((v) => v >= 2 ** 31).length / out.length;
    expect(odd).toBeGreaterThan(0.48);
    expect(odd).toBeLessThan(0.52);
    expect(high).toBeGreaterThan(0.48);
    expect(high).toBeLessThan(0.52);
  });
});

describe('randomSeed', () => {
  it('returns varied uint32 integers', () => {
    const seeds = Array.from({ length: 2000 }, () => randomSeed());
    for (const s of seeds) expect(Number.isInteger(s) && s >= 0 && s < UINT32).toBe(true);
    expect(new Set(seeds).size).toBeGreaterThanOrEqual(1998);
    expect(seeds.some((s) => s >= 2 ** 31)).toBe(true); // the top bit is used (no int32 sign bug)
    expect(seeds.some((s) => s % 2 === 1)).toBe(true);
    expect(seeds.some((s) => s % 2 === 0)).toBe(true);
  });

  it('assembles the uint32 from crypto.getRandomValues bytes', () => {
    const getRandomValues = vi.fn(<T extends ArrayBufferView | null>(arr: T): T => {
      new Uint8Array((arr as unknown as Uint8Array).buffer).set([0x12, 0x34, 0x56, 0xf8]);
      return arr;
    });
    vi.stubGlobal('crypto', { getRandomValues });
    expect(randomSeed()).toBe(0x123456f8);
    expect(getRandomValues).toHaveBeenCalledTimes(1);
  });

  it('falls back when crypto is unavailable or throws', () => {
    vi.stubGlobal('crypto', undefined);
    const a = randomSeed();
    expect(Number.isInteger(a) && a >= 0 && a < UINT32).toBe(true);
    vi.stubGlobal('crypto', {
      getRandomValues: () => {
        throw new Error('blocked');
      },
    });
    const b = randomSeed();
    expect(Number.isInteger(b) && b >= 0 && b < UINT32).toBe(true);
  });
});

describe('shuffle', () => {
  const xs = Array.from({ length: 20 }, (_, i) => i);

  it('returns a permutation in a new array and leaves the input untouched', () => {
    const input = Object.freeze([...xs]);
    const out = shuffle(input, makeRng(3));
    expect(out).not.toBe(input);
    expect([...out].sort((a, b) => a - b)).toEqual(xs);
    expect(input).toEqual(xs);
  });

  it('handles empty and single-element arrays (still a copy)', () => {
    const empty: number[] = [];
    const one = [{ k: 1 }];
    expect(shuffle(empty, makeRng(1))).toEqual([]);
    expect(shuffle(empty, makeRng(1))).not.toBe(empty);
    const s = shuffle(one, makeRng(1));
    expect(s).toEqual(one);
    expect(s).not.toBe(one);
    expect(s[0]).toBe(one[0]); // elements are moved, not cloned
  });

  it('is deterministic for the same rng seed and varies across seeds', () => {
    expect(shuffle(xs, makeRng(99))).toEqual(shuffle(xs, makeRng(99)));
    const orders = new Set(Array.from({ length: 20 }, (_, s) => shuffle(xs, makeRng(s)).join(',')));
    expect(orders.size).toBe(20);
  });

  it('draws every permutation with equal probability (3 items)', () => {
    const rng = makeRng(77);
    const counts = new Map<string, number>();
    for (let i = 0; i < 60_000; i++) {
      const key = shuffle(['a', 'b', 'c'], rng).join('');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    expect(counts.size).toBe(6);
    // chi-square, 5 degrees of freedom: 20.5 is the p = 0.001 critical value
    expect(chiSquare([...counts.values()])).toBeLessThan(20.5);
  });

  it('stays a valid permutation with degenerate rngs (0, just below 1, exactly 1)', () => {
    for (const value of [0, 1 - Number.EPSILON / 2, 1]) {
      const out = shuffle(xs, () => value);
      expect(out).toHaveLength(xs.length);
      expect([...out].sort((a, b) => a - b)).toEqual(xs);
    }
  });
});

describe('participant ids', () => {
  it('alphabet has 32 unique symbols without 0/O/1/I and equals the schema character class', () => {
    expect(PARTICIPANT_ID_ALPHABET).toHaveLength(32);
    expect(new Set(PARTICIPANT_ID_ALPHABET).size).toBe(32);
    for (const c of '0O1I') expect(PARTICIPANT_ID_ALPHABET).not.toContain(c);
    const candidates = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'];
    const schemaClass = candidates.filter((c) => PID_RE.test(`P-${c.repeat(6)}`));
    expect(schemaClass.sort().join('')).toBe([...PARTICIPANT_ID_ALPHABET].sort().join(''));
  });

  it('newParticipantId matches the schema pattern and uses the whole alphabet', () => {
    const ids = Array.from({ length: 2000 }, () => newParticipantId());
    for (const id of ids) {
      expect(id).toMatch(PID_RE);
      expect(isParticipantId(id)).toBe(true);
    }
    const used = new Set(ids.flatMap((id) => [...id.slice(2)]));
    expect(used.size).toBe(32);
    // 32^6 ≈ 1.07e9 ids: two or more collisions among 2000 is a ~1e-6 event
    expect(new Set(ids).size).toBeGreaterThanOrEqual(1999);
  });

  it('maps random bytes onto the alphabet without bias (byte & 31)', () => {
    vi.stubGlobal('crypto', {
      getRandomValues: <T extends ArrayBufferView | null>(arr: T): T => {
        (arr as unknown as Uint8Array).set([0, 1, 31, 32, 255, 200]);
        return arr;
      },
    });
    // 0→A 1→B 31→9 32→A 255→9 200(=8)→J
    expect(newParticipantId()).toBe('P-AB9A9J');
  });

  it('still produces valid ids without crypto', () => {
    vi.stubGlobal('crypto', undefined);
    for (let i = 0; i < 100; i++) expect(newParticipantId()).toMatch(PID_RE);
  });

  it('isParticipantId rejects malformed ids', () => {
    for (const bad of ['', 'P-ABCDE', 'P-ABCDEFG', 'p-ABCDEF', 'P-abcdef', 'P-ABCDE0', 'P-ABCDEO', 'P-ABCDE1', 'P-ABCDEI', 'X-ABCDEF', ' P-ABCDEF', 'P-ABCDEF ', 'PABCDEF']) {
      expect(isParticipantId(bad), bad).toBe(false);
    }
    expect(isParticipantId('P-2345ZZ')).toBe(true);
  });
});

describe('newId', () => {
  it('returns unique UUID v4 strings accepted by the schema Id pattern', () => {
    const ids = Array.from({ length: 1000 }, () => newId());
    for (const id of ids) {
      expect(id).toMatch(UUID_V4);
      expect(id).toMatch(ID_RE);
    }
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('prefers crypto.randomUUID when it exists', () => {
    vi.stubGlobal('crypto', { randomUUID: () => '0f8fad5b-d9cb-469f-a165-70867728950e', getRandomValues: realCrypto.getRandomValues.bind(realCrypto) });
    expect(newId()).toBe('0f8fad5b-d9cb-469f-a165-70867728950e');
  });

  it('builds a v4 UUID from getRandomValues when randomUUID is missing (insecure context)', () => {
    let fill = 0x00;
    vi.stubGlobal('crypto', {
      getRandomValues: <T extends ArrayBufferView | null>(arr: T): T => {
        (arr as unknown as Uint8Array).fill(fill);
        return arr;
      },
    });
    expect(newId()).toBe('00000000-0000-4000-8000-000000000000');
    fill = 0xff;
    expect(newId()).toBe('ffffffff-ffff-4fff-bfff-ffffffffffff');
    vi.stubGlobal('crypto', { getRandomValues: realCrypto.getRandomValues.bind(realCrypto) });
    const ids = Array.from({ length: 200 }, () => newId());
    for (const id of ids) expect(id).toMatch(UUID_V4);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('falls back to non-crypto randomness when crypto is missing or throws', () => {
    vi.stubGlobal('crypto', undefined);
    expect(newId()).toMatch(UUID_V4);
    vi.stubGlobal('crypto', {
      randomUUID: () => {
        throw new Error('not allowed');
      },
      getRandomValues: () => {
        throw new Error('not allowed');
      },
    });
    expect(newId()).toMatch(UUID_V4);
  });
});

describe('isoTime', () => {
  it('formats epoch milliseconds as UTC ISO 8601', () => {
    expect(isoTime(0)).toBe('1970-01-01T00:00:00.000Z');
    expect(isoTime(Date.UTC(2026, 9, 23, 12, 34, 56, 789))).toBe('2026-10-23T12:34:56.789Z');
    expect(isoTime(-1)).toBe('1969-12-31T23:59:59.999Z');
  });

  it('defaults to now', () => {
    const before = Date.now();
    const t = Date.parse(isoTime());
    expect(t).toBeGreaterThanOrEqual(before);
    expect(t).toBeLessThanOrEqual(Date.now());
  });

  it('never throws: invalid times fall back to now', () => {
    for (const bad of [Number.NaN, Infinity, -Infinity, 1e20]) {
      const before = Date.now();
      const t = Date.parse(isoTime(bad));
      expect(t).toBeGreaterThanOrEqual(before);
      expect(t).toBeLessThanOrEqual(Date.now());
    }
  });

  it('fits the schema IsoTime length limits', () => {
    const { minLength, maxLength } = recordsSchema.$defs.IsoTime;
    for (const t of [0, Date.now(), 8.64e15, -8.64e15]) {
      const s = isoTime(t);
      expect(s.length).toBeGreaterThanOrEqual(minLength);
      expect(s.length).toBeLessThanOrEqual(maxLength);
    }
  });
});

describe('roundOrNull', () => {
  it('rounds to the requested number of decimals', () => {
    expect(roundOrNull(1.23456, 2)).toBe(1.23);
    expect(roundOrNull(1.23456, 4)).toBe(1.2346);
    expect(roundOrNull(123.456789, 0)).toBe(123);
    expect(roundOrNull(42, 3)).toBe(42);
    expect(roundOrNull(9.80665123456, ROUNDING.acc)).toBe(9.8067);
    expect(roundOrNull(-12.3456789, ROUNDING.rr)).toBe(-12.346);
  });

  it('passes null, undefined and non-finite values through as null', () => {
    for (const v of [null, undefined, Number.NaN, Infinity, -Infinity]) expect(roundOrNull(v, 3)).toBeNull();
  });

  it('never returns -0 (JSON and Object.is safe)', () => {
    expect(Object.is(roundOrNull(-0, 3), 0)).toBe(true);
    expect(Object.is(roundOrNull(-0.00001, 4), 0)).toBe(true);
    expect(Object.is(roundOrNull(-0.4, 0), 0)).toBe(true);
    expect(Object.is(roundOrNull(0, 2), 0)).toBe(true);
    expect(JSON.stringify([roundOrNull(-0.00001, 4)])).toBe('[0]');
  });

  it('rounds halves away from zero, symmetrically', () => {
    expect(roundOrNull(0.5, 0)).toBe(1);
    expect(roundOrNull(-0.5, 0)).toBe(-1);
    expect(roundOrNull(2.5, 0)).toBe(3);
    expect(roundOrNull(-2.5, 0)).toBe(-3);
    expect(roundOrNull(0.125, 2)).toBe(0.13); // 0.125 is exact in binary
    expect(roundOrNull(-0.125, 2)).toBe(-0.13);
    const rng = makeRng(5);
    for (let i = 0; i < 10_000; i++) {
      const v = (rng() - 0.5) * 200;
      const r = roundOrNull(v, 3) as number;
      expect(Math.abs(r - v)).toBeLessThanOrEqual(0.0005 + 1e-12);
      expect((roundOrNull(-v, 3) as number) + r).toBe(0);
    }
  });

  it('leaves huge and tiny magnitudes sane (no overflow to Infinity)', () => {
    expect(roundOrNull(1e300, 4)).toBe(1e300);
    expect(roundOrNull(-1e300, 4)).toBe(-1e300);
    expect(roundOrNull(Number.MAX_VALUE, 6)).toBe(Number.MAX_VALUE);
    expect(roundOrNull(2 ** 60 + 0.5, 2)).toBe(2 ** 60);
    expect(roundOrNull(5e-324, 4)).toBe(0);
    expect(roundOrNull(1e-7, 6)).toBe(0);
  });
});

describe('scoreEhi (EHI short form)', () => {
  const ANSWERS = recordsSchema.$defs.EhiAnswer.enum as EhiAnswer[];
  const form = (w: EhiAnswer, t: EhiAnswer, b: EhiAnswer, s: EhiAnswer): EhiShortForm => ({
    writing: w,
    throwing: t,
    toothbrush: b,
    spoon: s,
  });

  it('items and answer scale match the schema', () => {
    expect([...EHI_ITEMS].sort()).toEqual([...recordsSchema.$defs.EhiShortForm.required].sort());
    expect(Object.keys(EHI_ANSWER_SCORES).sort()).toEqual([...ANSWERS].sort());
  });

  it('scores the textbook examples', () => {
    expect(scoreEhi(form('always_right', 'always_right', 'always_right', 'always_right'))).toEqual({ lq: 100, cls: 'right' });
    expect(scoreEhi(form('always_left', 'always_left', 'always_left', 'always_left'))).toEqual({ lq: -100, cls: 'left' });
    expect(scoreEhi(form('both', 'both', 'both', 'both'))).toEqual({ lq: 0, cls: 'mixed' });
    // "usually right" on every item is LQ 50: mixed under Veale's cut-offs
    expect(scoreEhi(form('usually_right', 'usually_right', 'usually_right', 'usually_right'))).toEqual({ lq: 50, cls: 'mixed' });
    expect(scoreEhi(form('always_right', 'always_right', 'usually_right', 'both'))).toEqual({ lq: 62.5, cls: 'right' });
    expect(scoreEhi(form('always_left', 'always_left', 'usually_left', 'both'))).toEqual({ lq: -62.5, cls: 'left' });
    expect(scoreEhi(form('always_right', 'usually_left', 'both', 'usually_right'))).toEqual({ lq: 25, cls: 'mixed' });
  });

  it('LQ is the item mean for all 625 answer combinations, with consistent classes', () => {
    let n = 0;
    for (const w of ANSWERS) for (const t of ANSWERS) for (const b of ANSWERS) for (const s of ANSWERS) {
      const { lq, cls } = scoreEhi(form(w, t, b, s));
      const mean = (EHI_ANSWER_SCORES[w] + EHI_ANSWER_SCORES[t] + EHI_ANSWER_SCORES[b] + EHI_ANSWER_SCORES[s]) / 4;
      expect(lq).toBe(mean);
      expect(Number.isInteger(lq / 12.5)).toBe(true);
      expect(lq).toBeGreaterThanOrEqual(-100);
      expect(lq).toBeLessThanOrEqual(100);
      expect(cls).toBe(lq >= 61 ? 'right' : lq <= -61 ? 'left' : 'mixed');
      n++;
    }
    expect(n).toBe(625);
  });

  it('cut-offs: LQ ≥ 61 right, ≤ −61 left, otherwise mixed', () => {
    expect(ehiClassFromLq(61)).toBe('right');
    expect(ehiClassFromLq(60.99)).toBe('mixed');
    expect(ehiClassFromLq(-60.99)).toBe('mixed');
    expect(ehiClassFromLq(-61)).toBe('left');
    expect(ehiClassFromLq(0)).toBe('mixed');
    expect(ehiClassFromLq(100)).toBe('right');
    expect(ehiClassFromLq(-100)).toBe('left');
  });

  it('throws on answers outside the scale instead of returning NaN', () => {
    const bad = { ...form('both', 'both', 'both', 'both'), spoon: 'sometimes' } as unknown as EhiShortForm;
    expect(() => scoreEhi(bad)).toThrow(/spoon/);
    const proto = { ...form('both', 'both', 'both', 'both'), writing: 'toString' } as unknown as EhiShortForm;
    expect(() => scoreEhi(proto)).toThrow(/writing/);
    const missing = { writing: 'both', throwing: 'both', toothbrush: 'both' } as unknown as EhiShortForm;
    expect(() => scoreEhi(missing)).toThrow(/spoon/);
  });
});
