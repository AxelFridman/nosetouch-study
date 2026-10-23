/**
 * Small utilities: seeded RNG, ids, time, rounding, EHI scoring.
 * DOM-free: runs in browsers (secure or not) and in Node 24 (tests, dataset synthesis).
 * IMPLEMENTED BY: protocol agent.
 */
import type { EhiAnswer, EhiClass, EhiShortForm } from './types';

/** mulberry32: deterministic uint32-seeded PRNG returning floats in [0, 1). */
export function makeRng(seed: number): () => number {
  // Seeds are taken modulo 2^32 (negative / oversized seeds wrap, NaN → 0).
  let a = seed >>> 0;
  return () => {
    // `| 0` keeps the state an exact int32. The textbook version lets it grow as a float, which
    // gives identical outputs until ~5 million calls and then silently loses precision.
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Derives an independent uint32 seed from (seed, salt) with the murmur3 finalizer, so one session
 * seed can feed several unrelated random streams (plan order, per-trial foreperiods, ...). For a
 * fixed salt it is a bijection: distinct seeds never collide.
 */
export function mixSeed(seed: number, salt: number): number {
  let h = ((seed >>> 0) ^ Math.imul(salt | 0, 0x9e3779b9)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

/** The global Web Crypto object if present (browsers and Node ≥ 19 both expose it). */
function cryptoApi(): Crypto | undefined {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  return c && typeof c === 'object' ? c : undefined;
}

/**
 * n random bytes. crypto.getRandomValues works in insecure contexts too (unlike randomUUID); the
 * Math.random fallback only exists for exotic runtimes: ids must be unique, not secret.
 */
function randomBytes(n: number): Uint8Array {
  const out = new Uint8Array(n);
  const c = cryptoApi();
  if (c && typeof c.getRandomValues === 'function') {
    try {
      c.getRandomValues(out);
      return out;
    } catch {
      // fall through to the non-crypto fallback
    }
  }
  for (let i = 0; i < n; i++) out[i] = Math.floor(Math.random() * 256);
  return out;
}

/** Cryptographically random uint32 (crypto.getRandomValues), for new session seeds. */
export function randomSeed(): number {
  const b = randomBytes(4);
  return ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) >>> 0;
}

/** Fisher–Yates shuffle (returns a new array) using the given rng. */
export function shuffle<T>(items: readonly T[], rng: () => number): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    // min() guards against a custom rng that returns exactly 1
    const j = Math.min(i, Math.floor(rng() * (i + 1)));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/** 32 symbols without the look-alikes 0/O and 1/I (participants may have to read the id aloud). */
export const PARTICIPANT_ID_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const PARTICIPANT_ID_RE = /^P-[A-HJ-NP-Z2-9]{6}$/; // = schema $defs.ParticipantId

/** "P-" + 6 chars from "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" (crypto random). */
export function newParticipantId(): string {
  let id = 'P-';
  // 256 is a multiple of 32, so `byte & 31` is unbiased. 32^6 ≈ 1.07e9 possible ids.
  for (const byte of randomBytes(6)) id += PARTICIPANT_ID_ALPHABET[byte & 31];
  return id;
}

/** true for a well-formed participant id (same pattern as the schema). */
export function isParticipantId(s: string): boolean {
  return PARTICIPANT_ID_RE.test(s);
}

/** UUID v4 (crypto.randomUUID when available, else getRandomValues-based). */
export function newId(): string {
  const c = cryptoApi();
  // randomUUID only exists in secure contexts (HTTPS / localhost), e.g. not on a plain-HTTP LAN.
  if (c && typeof c.randomUUID === 'function') {
    try {
      return c.randomUUID();
    } catch {
      // fall through
    }
  }
  const b = randomBytes(16);
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // RFC 4122 variant
  const hex = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const MAX_DATE_MS = 8.64e15; // ECMAScript Date range limit

/**
 * ISO 8601 string (UTC) for an epoch-ms time (default: now). A non-finite or out-of-range input
 * also yields "now" instead of throwing: a bad clock value must never cost a recorded trial.
 */
export function isoTime(epochMs?: number): string {
  const ok = typeof epochMs === 'number' && Number.isFinite(epochMs) && Math.abs(epochMs) <= MAX_DATE_MS;
  return new Date(ok ? (epochMs as number) : Date.now()).toISOString();
}

/**
 * Rounds to `decimals` (≥ 0) places; passes null/undefined/non-finite through as null.
 * Halves round away from zero (symmetric, so rounding adds no sign bias) and -0 becomes 0.
 */
export function roundOrNull(v: number | null | undefined, decimals: number): number | null {
  if (v === null || v === undefined || !Number.isFinite(v)) return null;
  const f = 10 ** decimals;
  const scaled = Math.abs(v) * f;
  // Past 2^52 a double has no fractional digits left at this scale (and v·f could overflow).
  if (!(scaled < 2 ** 52)) return v === 0 ? 0 : v;
  const r = (Math.sign(v) * Math.round(scaled)) / f;
  return r === 0 ? 0 : r;
}

/** The four EHI-SF items, in questionnaire order. */
export const EHI_ITEMS: readonly (keyof EhiShortForm)[] = ['writing', 'throwing', 'toothbrush', 'spoon'];

/** Item scores of the EHI-SF answer scale. */
export const EHI_ANSWER_SCORES: Readonly<Record<EhiAnswer, number>> = {
  always_right: 100,
  usually_right: 50,
  both: 0,
  usually_left: -50,
  always_left: -100,
};

/** Veale (2014) cut-offs: LQ ≥ 61 right, LQ ≤ −61 left, otherwise mixed. */
export function ehiClassFromLq(lq: number): EhiClass {
  if (lq >= 61) return 'right';
  if (lq <= -61) return 'left';
  return 'mixed';
}

/**
 * Edinburgh Handedness Inventory – Short Form (Veale, 2014): 4 items (writing, throwing,
 * toothbrush, spoon); always right = 100, usually right = 50, both = 0, usually left = −50,
 * always left = −100; LQ = mean. Class: LQ ≥ 61 right, LQ ≤ −61 left, otherwise mixed.
 * LQ is always a multiple of 12.5 in [−100, 100]. Throws on an answer outside the scale (only
 * possible with untyped/corrupted input), rather than storing a NaN quotient.
 */
export function scoreEhi(sf: EhiShortForm): { lq: number; cls: EhiClass } {
  let sum = 0;
  for (const item of EHI_ITEMS) {
    const answer = sf[item];
    const score = Object.prototype.hasOwnProperty.call(EHI_ANSWER_SCORES, answer) ? EHI_ANSWER_SCORES[answer] : undefined;
    if (score === undefined) throw new Error(`scoreEhi: invalid answer for "${item}": ${String(answer)}`);
    sum += score;
  }
  const lq = sum / EHI_ITEMS.length;
  return { lq, cls: ehiClassFromLq(lq) };
}
