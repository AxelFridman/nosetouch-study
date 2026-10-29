/**
 * TEST-ONLY helpers of the pipeline tests (never imported by the app): loading the golden fixtures written by
 * analysis/stage3 (format 'nosetouch-fixture-1', reference 'nosetouch-ref-1') and comparing float arrays.
 *
 *  - committed, synthetic: app/src/predict/__fixtures__/*.json.gz (+ index.json)
 *  - local only, real recordings: analysis/data/stage3_fixtures/ (git-ignored; absent in CI, so the tests that
 *    read it are skipped there). Keyed by trial key; never print anything but counts and numbers from them.
 */
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import type { BrowserEngine, StreamNormalization } from '../types';

export const SYNTH_DIR = new URL('../__fixtures__/', import.meta.url);
export const REAL_DIR = new URL('../../../../analysis/data/stage3_fixtures/', import.meta.url);

export type Num = number | null;

export interface FixtureSegment {
  seg: number;
  onset_ms: number;
  end_ms: number;
  decided_ms: number;
  canonical_ok: boolean;
  crop: { start_ms: number; stop_ms: number; n?: number; t0_ms?: number; i_on?: number; i_end?: number };
  gravity?: { method: string; pre_n: number; post_n: number; closure_deg: Num; magnitude: Num };
  /** (17, 64) float32 values */
  window?: Num[][];
}

export interface Fixture {
  id: string;
  kind: string;
  engine: BrowserEngine;
  normalization: StreamNormalization;
  motion: {
    t: number[];
    acc_x: Num[];
    acc_y: Num[];
    acc_z: Num[];
    accg_x: Num[];
    accg_y: Num[];
    accg_z: Num[];
    rr_alpha: Num[];
    rr_beta: Num[];
    rr_gamma: Num[];
  };
  reference: {
    engine: string;
    sign: number;
    n_raw: number;
    n_grid: number;
    repeats_dropped: number;
    normalization: StreamNormalization;
    stream: { t: number[]; f: Num[][]; acc: Num[][]; gyro: Num[][]; valid: boolean[] };
    segments: FixtureSegment[];
  };
}

export function fixturesPresent(dir: URL): boolean {
  return existsSync(new URL('index.json', dir));
}

export function loadFixtures(dir: URL): Fixture[] {
  const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8')) as { cases: { file: string }[] };
  return index.cases.map((c) => JSON.parse(gunzipSync(readFileSync(new URL(c.file, dir))).toString('utf8')) as Fixture);
}

export function num(v: Num | undefined): number {
  return v === null || v === undefined ? NaN : v;
}

/** (n, 3) rows with null -> Float64Array row-major */
export function rowsOf(r: Num[][]): Float64Array {
  const out = new Float64Array(3 * r.length);
  r.forEach((row, i) => {
    out[3 * i] = num(row[0]);
    out[3 * i + 1] = num(row[1]);
    out[3 * i + 2] = num(row[2]);
  });
  return out;
}

/** Running worst-case of a comparison (absolute difference, NaN-pattern mismatches, non-bitwise values). */
export class Diff {
  max = 0;
  n = 0;
  nanMismatch = 0;
  notBitwise = 0;
  add(a: number, b: number): void {
    this.n++;
    const an = Number.isNaN(a);
    const bn = Number.isNaN(b);
    if (an || bn) {
      if (an !== bn) this.nanMismatch++;
      return;
    }
    if (!Object.is(a, b)) this.notBitwise++;
    const d = Math.abs(a - b);
    if (d > this.max) this.max = d;
  }
  addAll(a: ArrayLike<number>, b: ArrayLike<number>): void {
    if (a.length !== b.length) throw new RangeError(`length ${a.length} != ${b.length}`);
    for (let i = 0; i < a.length; i++) this.add(a[i], b[i]);
  }
  merge(o: Diff): void {
    this.max = Math.max(this.max, o.max);
    this.n += o.n;
    this.nanMismatch += o.nanMismatch;
    this.notBitwise += o.notBitwise;
  }
  toString(): string {
    return `max|d|=${this.max.toExponential(2)} non-bitwise=${this.notBitwise}/${this.n} NaN-mismatch=${this.nanMismatch}`;
  }
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** Distance in float32 units in the last place between two float32 values (both finite or both NaN -> 0). */
export function ulp32(a: number, b: number): number {
  if (Number.isNaN(a) && Number.isNaN(b)) return 0;
  if (Number.isNaN(a) || Number.isNaN(b)) return Infinity;
  const ord = (x: number): number => {
    f32[0] = x;
    const u = u32[0];
    return u & 0x80000000 ? -(u & 0x7fffffff) : u;
  };
  return Math.abs(ord(a) - ord(b));
}
