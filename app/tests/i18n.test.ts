/**
 * i18n completeness: the Hebrew string tree must mirror the English one.
 *
 * Generic on purpose — it walks whatever namespaces and keys exist (the UI screens keep adding
 * strings), checking: identical key structure (and array lengths), no empty strings, and that
 * every function returns a non-empty string in BOTH languages. Function parameters are untyped at
 * runtime, so sample arguments are discovered: a coordinate search over numbers, every enum value
 * of the schema / app unions, every key of the English tree (enum-keyed lookups), two arrays, an
 * object whose properties are strings and a permissive "any" value. Each argument is then varied
 * over that pool to reach other branches (left/right, 1 vs many, ...): wherever English yields
 * text, Hebrew must too, and a 2-digit number shown by English must also be shown by Hebrew.
 */
import { describe, expect, it } from 'vitest';
import { common as enCommon } from '../src/i18n/en/common';
import { en } from '../src/i18n/en';
import { he } from '../src/i18n/he';
import type { Hand, OrientationId } from '../src/core/types';
import { recordsSchema } from './helpers/schema';

type Tree = Record<string, unknown>;
type Fn = (...args: unknown[]) => unknown;
type Kind = 'string' | 'function' | 'array' | 'object' | 'number' | 'boolean' | 'null' | 'undefined' | 'other';

const EN = en as unknown as Tree;
const HE = he as unknown as Tree;
const MAX_DEPTH = 12;
const MAX_ISSUES_PER_FUNCTION = 3;

function kindOf(v: unknown): Kind {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  const t = typeof v;
  if (t === 'string' || t === 'function' || t === 'number' || t === 'boolean' || t === 'undefined') return t;
  return t === 'object' ? 'object' : 'other';
}

// ───────────────────────────── structure ─────────────────────────────

/** Key/kind/array-length differences and empty strings between two string trees. */
function structureIssues(enNode: unknown, heNode: unknown, path: string, out: string[] = [], depth = 0): string[] {
  if (depth > MAX_DEPTH) {
    out.push(`${path}: nested deeper than ${MAX_DEPTH} levels (cycle?)`);
    return out;
  }
  const ke = kindOf(enNode);
  const kh = kindOf(heNode);
  if (ke !== kh) {
    out.push(`${path}: en is ${ke}, he is ${kh}`);
    return out;
  }
  switch (ke) {
    case 'string':
      if (!(enNode as string).trim()) out.push(`${path}: empty string in en`);
      if (!(heNode as string).trim()) out.push(`${path}: empty string in he`);
      break;
    case 'array': {
      const a = enNode as unknown[];
      const b = heNode as unknown[];
      if (a.length !== b.length) out.push(`${path}: en has ${a.length} items, he has ${b.length}`);
      for (let i = 0; i < Math.min(a.length, b.length); i++) structureIssues(a[i], b[i], `${path}[${i}]`, out, depth + 1);
      break;
    }
    case 'object': {
      const a = enNode as Tree;
      const b = heNode as Tree;
      for (const k of Object.keys(a)) if (!Object.hasOwn(b, k)) out.push(`${path}.${k}: missing in he`);
      for (const k of Object.keys(b)) if (!Object.hasOwn(a, k)) out.push(`${path}.${k}: only in he (missing in en)`);
      for (const k of Object.keys(a)) if (Object.hasOwn(b, k)) structureIssues(a[k], b[k], `${path}.${k}`, out, depth + 1);
      break;
    }
    case 'undefined':
    case 'other':
      out.push(`${path}: unsupported value (${ke})`);
      break;
    default:
      break; // function / number / boolean / null: matching kinds are all we can say here
  }
  return out;
}

// ───────────────────────────── functions ─────────────────────────────

/** Survives any use: property access, calls, iteration, arithmetic and string conversion. */
function makeAny(): unknown {
  const proxy: unknown = new Proxy(function any() {}, {
    get(_target, prop) {
      if (prop === Symbol.toPrimitive) return (hint: string) => (hint === 'number' ? 41 : '41');
      if (prop === Symbol.iterator) {
        return function* () {
          yield proxy;
          yield proxy;
        };
      }
      if (prop === 'length') return 2;
      if (prop === 'toString' || prop === 'toLocaleString' || prop === 'valueOf') return () => '41';
      return typeof prop === 'symbol' ? undefined : proxy;
    },
    apply: () => proxy,
  });
  return proxy;
}
const ANY = makeAny();
/** An object whose every property is `value` (option bags such as { name, email } or { done, total }). */
function makeBag(value: string | number): unknown {
  return new Proxy(
    {},
    {
      get: (_target, prop) =>
        prop === 'toString' || prop === 'valueOf' ? () => String(value) : typeof prop === 'symbol' ? undefined : value,
    },
  );
}
const TEXTY = makeBag('sample');
const NUMBERY = makeBag(37);
const DATE = new Date(Date.UTC(2026, 0, 2, 3, 4, 5));

/** Runtime unions of core/types.ts and app/state.ts that the JSON schema does not list. */
const APP_UNIONS: string[][] = [
  // TrialMessage, TrialPhase, SensorError['code'], UploadMode, OutboxItem['status'], ScreenFacing,
  // toast kind, ScreenId
  ['hold_still', 'too_early', 'go', 'moving', 'hold_at_nose', 'done', 'no_movement', 'too_long', 'sensor_stalled', 'return_arm', 'wrong_grip'],
  ['waiting_still', 'foreperiod', 'armed', 'moving', 'post_roll', 'returning', 'done'],
  ['not_supported', 'permission_denied', 'insecure_context', 'no_events', 'sensor_error'],
  ['apps_script', 'json_post', 'none'],
  ['pending', 'sent', 'failed'],
  ['toward', 'away', 'up'],
  ['info', 'ok', 'warn', 'error'],
  ['loading', 'welcome', 'profile', 'sensors', 'setup', 'session', 'done', 'data', 'settings', 'sandbox', 'about'],
];

function schemaEnumValues(node: unknown, out: Set<string>): Set<string> {
  if (Array.isArray(node)) node.forEach((n) => schemaEnumValues(n, out));
  else if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) {
      if (k === 'enum' && Array.isArray(v)) v.forEach((x) => typeof x === 'string' && out.add(x));
      else schemaEnumValues(v, out);
    }
  }
  return out;
}

function treeKeys(node: unknown, out: Set<string>, depth = 0): Set<string> {
  if (depth > MAX_DEPTH || !node || typeof node !== 'object') return out;
  for (const [k, v] of Object.entries(node)) {
    if (!Array.isArray(node)) out.add(k);
    treeKeys(v, out, depth + 1);
  }
  return out;
}

const NUMBERS = [37, 59, 1, 2, 0];
const POOL: unknown[] = [
  ...NUMBERS,
  ...new Set([
    ...schemaEnumValues(recordsSchema, new Set()),
    ...APP_UNIONS.flat(),
    ...treeKeys(EN, new Set()),
    'sample text',
    DATE.toISOString(),
  ]),
  ['alpha', 'beta'],
  [ANY, ANY],
  TEXTY,
  NUMBERY,
  DATE,
  ANY,
];

type Outcome = { rank: 0 | 1 | 2; why: string } | { rank: 3; text: string };
const GARBAGE = /\bundefined\b|\bNaN\b|\[object \w+\]|Invalid Date/;

/** rank 0 threw, 1 not a non-empty string, 2 string with garbage (undefined/NaN/...), 3 clean text. */
function run(fn: Fn, args: unknown[]): Outcome {
  try {
    const raw = fn(...args);
    // the "any" proxy comes back from call chains on it (p.items.map(...).join()): with real
    // arguments such a chain yields whatever the real methods return, so count it as text
    const r = raw === ANY ? String(raw) : raw;
    if (typeof r !== 'string') return { rank: 1, why: `returned ${kindOf(r)}` };
    if (!r.trim()) return { rank: 1, why: 'returned an empty string' };
    if (GARBAGE.test(r)) return { rank: 2, why: `returned "${r}"` };
    return { rank: 3, text: r };
  } catch (e) {
    return { rank: 0, why: `threw "${e instanceof Error ? e.message : String(e)}"` };
  }
}

function labelOne(a: unknown): string {
  if (a === ANY) return '<any>';
  if (a === TEXTY) return '<object of strings>';
  if (a === NUMBERY) return '<object of numbers>';
  if (a === DATE) return `Date(${DATE.toISOString()})`;
  if (Array.isArray(a)) return `[${a.map(labelOne).join(', ')}]`;
  return typeof a === 'string' ? JSON.stringify(a) : String(a);
}

function label(args: unknown[]): string {
  return args.map(labelOne).join(', ');
}

/** Coordinate search for arguments that make `fn` return clean text. */
function findArgs(fn: Fn): { args: unknown[]; best: Outcome } {
  const k = Math.max(fn.length, 1);
  let args: unknown[] = Array.from({ length: k }, (_, i) => 37 + 22 * i);
  let best = run(fn, args);
  for (let pass = 0; pass < 3 && best.rank < 3; pass++) {
    for (let j = 0; j < k && best.rank < 3; j++) {
      for (const v of POOL) {
        const candidate = args.slice();
        candidate[j] = v;
        const o = run(fn, candidate);
        if (o.rank > best.rank) {
          args = candidate;
          best = o;
          if (o.rank === 3) break;
        }
      }
    }
  }
  return { args, best };
}

function checkFunction(enFn: Fn, heFn: Fn, path: string, out: string[]): void {
  const { args, best } = findArgs(enFn);
  if (best.rank !== 3) {
    out.push(`${path}: no sample arguments make the English function return text (best attempt ${label(args)}: ${best.why})`);
    return;
  }
  const probes: unknown[][] = [args];
  for (let j = 0; j < args.length; j++) {
    for (const v of POOL) {
      if (v === args[j]) continue;
      const p = args.slice();
      p[j] = v;
      probes.push(p);
    }
  }
  let n = 0;
  for (const p of probes) {
    if (n >= MAX_ISSUES_PER_FUNCTION) break;
    const e = run(enFn, p);
    if (e.rank !== 3) continue; // not a meaningful input for this function
    const h = run(heFn, p);
    if (h.rank !== 3) {
      out.push(`${path}(${label(p)}): he ${h.why}; en returned "${e.text}"`);
      n++;
      continue;
    }
    for (const a of p) {
      // 2-digit numbers never appear by accident and no language spells them out
      if (typeof a === 'number' && a >= 10 && e.text.includes(String(a)) && !h.text.includes(String(a))) {
        out.push(`${path}(${label(p)}): he does not show ${a}: en "${e.text}", he "${h.text}"`);
        n++;
      }
    }
  }
}

function isTree(v: unknown): v is Tree {
  return kindOf(v) === 'object';
}

/** Runs checkFunction on every function present in both trees (structure is checked separately). */
function functionIssues(enNode: unknown, heNode: unknown, path: string, out: string[] = [], depth = 0): string[] {
  if (depth > MAX_DEPTH) return out;
  if (typeof enNode === 'function') {
    if (typeof heNode === 'function') checkFunction(enNode as Fn, heNode as Fn, path, out);
  } else if (Array.isArray(enNode) && Array.isArray(heNode)) {
    for (let i = 0; i < Math.min(enNode.length, heNode.length); i++) functionIssues(enNode[i], heNode[i], `${path}[${i}]`, out, depth + 1);
  } else if (isTree(enNode) && isTree(heNode)) {
    for (const k of Object.keys(enNode)) {
      if (Object.hasOwn(heNode, k)) functionIssues(enNode[k], heNode[k], `${path}.${k}`, out, depth + 1);
    }
  }
  return out;
}

// ───────────────────────────── helpers for the self-checks ─────────────────────────────

/** Deep copy of a string tree; functions are shared. */
function cloneTree<T>(node: T): T {
  if (Array.isArray(node)) return node.map(cloneTree) as T;
  if (isTree(node)) return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, cloneTree(v)])) as T;
  return node;
}

/** Paths of all object-held leaves (strings, functions, arrays, ...). */
function leafPaths(node: unknown, prefix: string[] = []): string[][] {
  if (!isTree(node)) return [];
  return Object.entries(node).flatMap(([k, v]) => (isTree(v) ? leafPaths(v, [...prefix, k]) : [[...prefix, k]]));
}

function withoutLeaf(tree: unknown, path: string[]): unknown {
  const copy = cloneTree(tree);
  let node = copy as Tree;
  for (const k of path.slice(0, -1)) node = node[k] as Tree;
  delete node[path[path.length - 1]];
  return copy;
}

// ───────────────────────────── the real trees ─────────────────────────────

const NAMESPACES = [...new Set([...Object.keys(EN), ...Object.keys(HE)])].sort();

describe('i18n: Hebrew mirrors English', () => {
  it('both languages define the same namespaces', () => {
    expect(Object.keys(HE).sort()).toEqual(Object.keys(EN).sort());
    expect(NAMESPACES.length).toBeGreaterThan(0);
  });

  for (const ns of NAMESPACES) {
    it(`${ns}: identical key structure and no empty strings`, () => {
      expect(structureIssues(EN[ns], HE[ns], ns)).toEqual([]);
    });

    it(`${ns}: every function returns a non-empty string in both languages`, () => {
      expect(functionIssues(EN[ns], HE[ns], ns)).toEqual([]);
    });

    it(`${ns}: deleting any Hebrew entry is detected`, () => {
      const leaves = leafPaths(HE[ns]);
      for (const path of leaves) {
        const issues = structureIssues(EN[ns], withoutLeaf(HE[ns], path), ns);
        expect(issues).toContain(`${[ns, ...path].join('.')}: missing in he`);
      }
    });
  }
});

describe('i18n checker self-test (synthetic trees)', () => {
  const enTree = {
    title: 'Title',
    group: { a: 'A', count: (n: number) => `${n} items` },
    list: ['one', 'two'],
  };
  /** A correct Hebrew tree with one edit applied. */
  const heTree = (edit: (t: Tree, group: Tree) => void = () => {}): Tree => {
    const t: Tree = {
      title: 'כותרת',
      group: { a: 'א', count: (n: number) => `${n} פריטים` },
      list: ['אחת', 'שתיים'],
    };
    edit(t, t.group as Tree);
    return t;
  };

  it('accepts a complete translation', () => {
    expect(structureIssues(enTree, heTree(), 'root')).toEqual([]);
    expect(functionIssues(enTree, heTree(), 'root')).toEqual([]);
  });

  it('FAILS on a missing Hebrew key (and on extra, empty, mismatched and short entries)', () => {
    const missing = heTree((_t, g) => delete g.a);
    expect(structureIssues(enTree, missing, 'root')).toEqual(['root.group.a: missing in he']);

    const extra = heTree((_t, g) => (g.b = 'ב'));
    expect(structureIssues(enTree, extra, 'root')).toEqual(['root.group.b: only in he (missing in en)']);

    const empty = heTree((t) => (t.title = '   '));
    expect(structureIssues(enTree, empty, 'root')).toEqual(['root.title: empty string in he']);
    expect(structureIssues({ ...enTree, title: '' }, heTree(), 'root')).toEqual(['root.title: empty string in en']);

    const kind = heTree((t) => (t.group = 'flat'));
    expect(structureIssues(enTree, kind, 'root')).toEqual(['root.group: en is object, he is string']);

    const short = heTree((t) => (t.list = ['אחת']));
    expect(structureIssues(enTree, short, 'root')).toEqual(['root.list: en has 2 items, he has 1']);

    const undef = heTree((t) => (t.title = undefined));
    expect(structureIssues(enTree, undef, 'root')).toEqual(['root.title: en is string, he is undefined']);
  });

  it('FAILS on Hebrew functions that return nothing, throw, drop the number or print garbage', () => {
    const emptyFn = heTree((_t, g) => (g.count = () => ''));
    expect(functionIssues(enTree, emptyFn, 'root')[0]).toMatch(/^root\.group\.count\(37\): he returned an empty string/);

    const throwing = heTree(
      (_t, g) =>
        (g.count = () => {
          throw new Error('boom');
        }),
    );
    expect(functionIssues(enTree, throwing, 'root')[0]).toMatch(/he threw "boom"/);

    const dropped = heTree((_t, g) => (g.count = () => 'פריטים'));
    expect(functionIssues(enTree, dropped, 'root')[0]).toMatch(/he does not show 37/);

    const garbage = heTree((_t, g) => (g.count = (n: { x: number }) => `${n.x} פריטים`));
    expect(functionIssues(enTree, garbage, 'root')[0]).toMatch(/he returned "undefined/);
  });

  it('FAILS when only one Hebrew branch is broken (arguments are varied)', () => {
    const enHand = { hold: (hand: Hand) => (hand === 'left' ? 'Left hand' : 'Right hand') };
    const heHand = { hold: (hand: Hand) => (hand === 'left' ? '' : 'יד ימין') };
    const issues = functionIssues(enHand, heHand, 'root');
    expect(issues.length).toBeGreaterThan(0);
    expect(issues[0]).toMatch(/root\.hold\("left"\): he returned an empty string/);
    const enPlural = { n: (n: number) => (n === 1 ? 'one trial' : `${n} trials`) };
    const hePlural = { n: (n: number) => (n === 1 ? '' : `${n} ניסיונות`) };
    expect(functionIssues(enPlural, hePlural, 'root')).toEqual([expect.stringMatching(/root\.n\(1\): he returned an empty string/)]);
  });

  it('FAILS when the English function cannot produce text at all', () => {
    const bad = {
      f: () => {
        throw new Error('always');
      },
    };
    expect(functionIssues(bad, bad, 'root')).toEqual([expect.stringMatching(/^root\.f: no sample arguments/)]);
  });

  it('finds arguments for object, enum-lookup and multi-parameter functions (no false alarms)', () => {
    const enFns = {
      progress: ({ done, total }: { done: number; total: number }) => `${done} of ${total}`,
      block: (hand: Hand, o: OrientationId) => `${enCommon.hands[hand]}: ${enCommon.orientations[o].label}`,
      list: (items: string[]) => items.map((s) => s.toUpperCase()).join(', '),
      seconds: (ms: number) => `${(ms / 1000).toFixed(1)} s`,
      none: () => 'static',
      day: (d: Date) => `Started ${d.toISOString().slice(0, 10)}`,
      mean: (p: { n: number }) => `mean ${p.n.toFixed(1)}`,
      labels: (p: { items: { label: string }[] }) => p.items.map((i) => i.label).join(', '),
    };
    const heFns = {
      progress: ({ done, total }: { done: number; total: number }) => `${done} מתוך ${total}`,
      block: (hand: Hand, o: OrientationId) => `${(HE.common as typeof enCommon).hands[hand]}: ${(HE.common as typeof enCommon).orientations[o].label}`,
      list: (items: string[]) => items.map((s) => s.toUpperCase()).join(', '),
      seconds: (ms: number) => `${(ms / 1000).toFixed(1)} שנ׳`,
      none: () => 'קבוע',
      day: (d: Date) => `התחיל ${d.toISOString().slice(0, 10)}`,
      mean: (p: { n: number }) => `ממוצע ${p.n.toFixed(1)}`,
      labels: (p: { items: { label: string }[] }) => p.items.map((i) => i.label).join(', '),
    };
    expect(functionIssues(enFns, heFns, 'root')).toEqual([]);
  });
});
