/**
 * The standardised stream of a GROWING raw buffer, computed incrementally and bit for bit equal to
 * streamFrom(raw, norm) on the same rows (incremental.test.ts checks every prefix of every fixture).
 *
 * Why it can be exact (PORT_SPEC 1.9, with the zero-phase filter made exact as well):
 *  - Repeats: an isolated repeat depends on the next row, so only the newest row is provisional, and a dropped
 *    newest row can only turn into a kept one (preprocess.py:589-591). Kept rows are therefore only appended.
 *  - Resampling: a grid value depends on the two finite samples around it. Every sample that arrives later has a
 *    larger time, so all grid values at or before a column's last finite sample are final; later grid points are
 *    NaN until the next sample arrives.
 *  - Low-pass: per finite run. Once a run holds 16 values its odd-extension pad (15 values) is fixed, so the
 *    forward pass continues from its saved state. The backward pass starts from the end of the right pad, which
 *    changes with every new value; it is recomputed from the end and stopped at the first checkpoint where its
 *    state equals, bit for bit, the state the previous pass had there: from that point on the recursion is the
 *    same deterministic computation on the same inputs, so the older outputs are already exact. (The difference
 *    decays like 0.668^distance, so this happens after about 100 samples; without it the pass runs to the start.)
 *
 * Fallback: rows whose time is not finite and increasing (never from a LiveBuffer) or the iOS quantisation switch
 * make the builder call streamFrom on the whole buffer instead.
 */
import type { RawMotion, Stream, StreamNormalization } from '../types';
import { FS_HZ, LOWPASS_PADLEN, MIN_ROWS, RAD2DEG } from './constants';
import { scaledZi, searchLeft, searchRight, sosStep, sosfiltfiltRun } from './numeric';
import { RR_MAPPINGS } from './normalize';
import { LOWPASS, dropIsolated, gridCount, gridTime, resampleAt, sameAsPrevious } from './preprocess';
import { PipelineError, streamFrom } from './standardize';

/** spacing of the saved backward-pass states (samples) */
const CHECKPOINT = 16;
const PAD = LOWPASS_PADLEN;
/** a run needs PAD + 1 values before its left pad is fixed */
const MIN_INCREMENTAL_RUN = PAD + 1;

export interface StreamBuilderStats {
  /** raw rows consumed */
  readonly rows: number;
  /** grid samples of the current stream */
  readonly grid: number;
  /** runs filtered from scratch (short runs, new runs, restarts) */
  readonly fullRuns: number;
  /** backward-pass steps of the incremental updates (the work that replaces a full pass) */
  readonly backwardSteps: number;
  /** updates that converged before the start of their run */
  readonly converged: number;
  /** true once the builder fell back to streamFrom on every update */
  readonly batchOnly: boolean;
}

export interface StreamBuilder {
  /**
   * The stream of `raw`, identical to streamFrom(raw, norm); null while fewer than 5 rows are kept (where
   * streamFrom throws). `raw` normally extends the previous call's buffer (LiveBuffer.raw() after more pushes);
   * anything else (fewer rows, other first or last rows) restarts the builder. The returned arrays are copies.
   */
  update(raw: RawMotion): Stream | null;
  reset(): void;
  readonly stats: StreamBuilderStats;
}

/** The incremental zero-phase filter state of one open run. */
interface RunCache {
  start: number;
  /** values forward-filtered so far (x[0..len)) */
  len: number;
  /** forward outputs over [left pad (PAD), x[0..len)) */
  fwd: F64;
  /** forward state after x[len - 1] */
  zf: F64;
  /** backward states after output k = c * CHECKPOINT, 4 per checkpoint */
  bwd: F64;
  nCk: number;
}

interface Column {
  tv: F64;
  xv: F64;
  len: number;
  /** resampled values on the grid (before the low-pass) */
  u: F64;
  /** filtered values */
  out: F64;
  /** grid points [0, done) are final */
  done: number;
  /** start of the run ending at done - 1 (finite there), else -1 */
  openStart: number;
  cache: RunCache | null;
}

type F64 = Float64Array<ArrayBuffer>;
type U8 = Uint8Array<ArrayBuffer>;

function growF(a: F64, need: number): F64 {
  if (need <= a.length) return a;
  let cap = Math.max(a.length, 16);
  while (cap < need) cap *= 2;
  const b = new Float64Array(cap);
  b.set(a);
  return b;
}

function growU8(a: U8, need: number): U8 {
  if (need <= a.length) return a;
  let cap = Math.max(a.length, 16);
  while (cap < need) cap *= 2;
  const b = new Uint8Array(cap);
  b.set(a);
  return b;
}

/** the values that identify the first k rows of a buffer (k >= 1) */
function signatureOf(raw: RawMotion, k: number): number[] {
  const i = k - 1;
  const row = (d: Float64Array): number[] => [d[3 * i], d[3 * i + 1], d[3 * i + 2]];
  return [raw.t[0], raw.t[k >> 1], raw.t[i], ...row(raw.acc.data), ...row(raw.accg.data), ...row(raw.rr.data)];
}

function sameValues(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, j) => Object.is(v, b[j]) || (Number.isNaN(v) && Number.isNaN(b[j])));
}

function newColumn(): Column {
  return {
    tv: new Float64Array(256),
    xv: new Float64Array(256),
    len: 0,
    u: new Float64Array(256),
    out: new Float64Array(256),
    done: 0,
    openStart: -1,
    cache: null,
  };
}

/**
 * A drop-in replacement for streamFrom (same signature, same numbers, same PipelineError when fewer than 5 rows
 * are kept) for a caller that passes a GROWING buffer on every tick, such as the session engine's
 * `pipeline.streamFrom(buffer.raw(), norm)`: backed by one StreamBuilder, so a call costs the new rows instead of
 * the whole buffer (about 0.4 ms against 14 ms at the 3-minute cap on a desktop). A different normalisation or a
 * buffer that does not extend the previous one restarts the builder, so any call sequence stays correct.
 */
export function createIncrementalStreamFrom(): (raw: RawMotion, norm: StreamNormalization) => Stream {
  let builder: StreamBuilder | null = null;
  let key = '';
  return (raw: RawMotion, norm: StreamNormalization): Stream => {
    const k = JSON.stringify(norm);
    if (builder === null || k !== key) {
      builder = createStreamBuilder(norm);
      key = k;
    }
    const s = builder.update(raw);
    if (s === null) throw new PipelineError('not_enough_data', `fewer than ${MIN_ROWS} motion samples`);
    return s;
  };
}

export function createStreamBuilder(norm: StreamNormalization): StreamBuilder {
  const sign = norm.gravitySign === -1 ? -1.0 : 1.0;
  const [rrSign, rrCols] = RR_MAPPINGS[norm.rrVerdict] ?? RR_MAPPINGS.w3c;
  const rrK = rrSign * (norm.rrUnits === 'rad_per_s' ? RAD2DEG : 1.0);
  const quantized = norm.quantizeIos && norm.engine === 'webkit';

  let m = 0; // raw rows consumed
  /** identifies the consumed rows: t[0], t[m >> 1], t[m - 1] and the 9 raw values of row m - 1 */
  let signature: number[] = [];
  let same = new Uint8Array(256);
  let dropped = new Uint8Array(256);
  let nDropped = 0;
  let nKept = 0;
  let keptFirstT = NaN;
  let keptLastT = NaN;
  let tg = new Float64Array(256);
  let ng = 0;
  let valid = new Uint8Array(256);
  let validDone = 0;
  let cols: Column[] = [];
  let batchOnly = quantized;
  const counters = { fullRuns: 0, backwardSteps: 0, converged: 0 };

  function init(): void {
    m = 0;
    signature = [];
    same = new Uint8Array(256);
    dropped = new Uint8Array(256);
    nDropped = 0;
    nKept = 0;
    keptFirstT = NaN;
    keptLastT = NaN;
    tg = new Float64Array(256);
    ng = 0;
    valid = new Uint8Array(256);
    validDone = 0;
    cols = Array.from({ length: 9 }, newColumn);
  }
  init();

  /** signed value of raw row i in stream column j (0-2 f = accg, 3-5 acc, 6-8 gyro), as normalize_motion computes it */
  function signedValue(raw: RawMotion, i: number, j: number): number {
    if (j < 3) return raw.accg.data[3 * i + j] * sign;
    if (j < 6) return raw.acc.data[3 * i + j - 3] * sign;
    return raw.rr.data[3 * i + rrCols[j - 6]] * rrK;
  }

  function appendKept(raw: RawMotion, i: number): void {
    const t = raw.t[i];
    if (nKept === 0) keptFirstT = t;
    keptLastT = t;
    nKept++;
    for (let j = 0; j < 9; j++) {
      const v = signedValue(raw, i, j);
      if (!Number.isFinite(v)) continue;
      const c = cols[j];
      c.tv = growF(c.tv, c.len + 1);
      c.xv = growF(c.xv, c.len + 1);
      c.tv[c.len] = t;
      c.xv[c.len] = v;
      c.len++;
    }
  }

  /** consume rows [m, n); false when the rows break the incremental contract */
  function consume(raw: RawMotion, n: number): boolean {
    same = growU8(same, n);
    dropped = growU8(dropped, n);
    for (let i = m; i < n; i++) {
      const ti = raw.t[i];
      if (!Number.isFinite(ti) || (i > 0 && !(ti > raw.t[i - 1]))) return false;
      same[i] = sameAsPrevious(raw, i) ? 1 : 0;
      // the previous row was dropped as an isolated repeat while it was the newest row: a repeat after it
      // makes it part of a constant run, which is kept (isolated[i-1] needs NOT same[i])
      if (i >= 1 && dropped[i - 1] === 1 && same[i] === 1) {
        dropped[i - 1] = 0;
        nDropped--;
        appendKept(raw, i - 1);
      }
      // the newest row, provisionally (same[i + 1] unknown = false)
      const isolated = same[i] === 1 && !(i >= 1 && same[i - 1] === 1);
      if (isolated && dropIsolated(raw.rr, i, norm.engine)) {
        dropped[i] = 1;
        nDropped++;
      } else {
        dropped[i] = 0;
        appendKept(raw, i);
      }
    }
    m = n;
    signature = signatureOf(raw, n);
    return true;
  }

  /** zero-phase filter of run [a, b) of column c into c.out, reusing the open-run cache when it is this run */
  function filterRun(c: Column, a: number, b: number): void {
    const L = b - a;
    const x = c.u;
    if (L < 3) {
      for (let i = a; i < b; i++) c.out[i] = x[i];
      c.cache = null;
      return;
    }
    if (L < MIN_INCREMENTAL_RUN) {
      sosfiltfiltRun(LOWPASS, x, a, L, c.out, PAD);
      c.cache = null;
      counters.fullRuns++;
      return;
    }
    let k0 = c.cache && c.cache.start === a ? c.cache : null;
    if (k0 && k0.len === L) return; // same run, same values: the outputs are current
    if (!k0) {
      k0 = { start: a, len: 0, fwd: new Float64Array(PAD + L + 64), zf: new Float64Array(4), bwd: new Float64Array(64), nCk: 0 };
      c.cache = k0;
      counters.fullRuns++;
    }
    const cache = k0;
    const prevLen = cache.len;
    const nCkPrev = cache.nCk;
    cache.fwd = growF(cache.fwd, PAD + L);
    const fwd = cache.fwd;
    // forward pass over the new values (and the left pad the first time)
    let z: Float64Array;
    if (prevLen === 0) {
      const x0 = x[a];
      const e0 = 2 * x0 - x[a + PAD];
      z = scaledZi(LOWPASS, e0);
      for (let i = 0; i < PAD; i++) fwd[i] = sosStep(LOWPASS, z, 2 * x0 - x[a + PAD - i]);
    } else {
      z = cache.zf.slice();
    }
    for (let k = prevLen; k < L; k++) fwd[PAD + k] = sosStep(LOWPASS, z, x[a + k]);
    cache.zf.set(z);
    cache.len = L;
    // right pad: forward from the saved state (not kept: it changes with every new value)
    const fr = new Float64Array(PAD);
    const xl = x[a + L - 1];
    for (let i = 0; i < PAD; i++) fr[i] = sosStep(LOWPASS, z, 2 * xl - x[a + L - 2 - i]);
    // backward pass from the end of the right pad
    const zb = scaledZi(LOWPASS, fr[PAD - 1]);
    for (let i = PAD - 1; i >= 0; i--) sosStep(LOWPASS, zb, fr[i]);
    const nCk = Math.floor((L - 1) / CHECKPOINT) + 1;
    cache.bwd = growF(cache.bwd, 4 * nCk);
    const bwd = cache.bwd;
    let steps = 0;
    for (let k = L - 1; k >= 0; k--) {
      c.out[a + k] = sosStep(LOWPASS, zb, fwd[PAD + k]);
      steps++;
      if (k % CHECKPOINT !== 0) continue;
      const q = 4 * (k / CHECKPOINT);
      if (k < prevLen && k / CHECKPOINT < nCkPrev && Object.is(zb[0], bwd[q]) && Object.is(zb[1], bwd[q + 1]) && Object.is(zb[2], bwd[q + 2]) && Object.is(zb[3], bwd[q + 3])) {
        counters.converged++;
        break;
      }
      bwd[q] = zb[0];
      bwd[q + 1] = zb[1];
      bwd[q + 2] = zb[2];
      bwd[q + 3] = zb[3];
    }
    cache.nCk = nCk;
    if (prevLen > 0) counters.backwardSteps += steps;
  }

  /** resample and filter column c for the grid [0, ng) */
  function updateColumn(c: Column): number {
    const from = c.done;
    c.u = growF(c.u, ng);
    c.out = growF(c.out, ng);
    let r = c.len ? searchLeft(c.tv, tg[from] ?? Infinity, 0, c.len) : 0;
    for (let i = from; i < ng; i++) {
      const tn = tg[i];
      while (r < c.len && c.tv[r] < tn) r++;
      const v = c.len ? resampleAt(c.tv, c.xv, c.len, tn, r) : NaN;
      c.u[i] = v;
      if (v !== v) c.out[i] = NaN;
    }
    const done = c.len ? Math.min(ng, searchRight(tg, c.tv[c.len - 1], 0, ng)) : 0;
    let cs = c.openStart;
    for (let i = from; i < done; i++) {
      if (Number.isFinite(c.u[i])) {
        if (cs < 0) cs = i;
      } else if (cs >= 0) {
        filterRun(c, cs, i); // closed by a final NaN
        c.cache = null;
        cs = -1;
      }
    }
    if (cs >= 0) filterRun(c, cs, done); // the open run (may grow)
    c.openStart = cs;
    c.done = done;
    return from;
  }

  function build(raw: RawMotion): Stream {
    const t = tg.slice(0, ng);
    const inter = (j0: number): Float64Array => {
      const d = new Float64Array(3 * ng);
      for (let c = 0; c < 3; c++) {
        const o = cols[j0 + c].out;
        for (let i = 0; i < ng; i++) d[3 * i + c] = o[i];
      }
      return d;
    };
    return {
      t,
      f: { n: ng, data: inter(0) },
      acc: { n: ng, data: inter(3) },
      gyro: { n: ng, data: inter(6) },
      valid: valid.slice(0, ng),
      fsHz: FS_HZ,
      engine: norm.engine,
      repeatsDropped: nDropped,
      tRaw: raw.t.slice(0, m),
      gravityMagnitude: null,
      baselineGravity: null,
    };
  }

  function batch(raw: RawMotion): Stream | null {
    try {
      return streamFrom(raw, norm);
    } catch (e) {
      if (e instanceof PipelineError && e.code === 'not_enough_data') return null;
      throw e;
    }
  }

  return {
    update(raw: RawMotion): Stream | null {
      const n = raw.t.length;
      if (batchOnly) return batch(raw);
      // a buffer that does not extend the consumed rows (shorter, cleared and refilled, another stream) restarts
      if (n < m || (m > 0 && !sameValues(signature, signatureOf(raw, m)))) init();
      if (n > m && !consume(raw, n)) {
        batchOnly = true;
        return batch(raw);
      }
      if (nKept < MIN_ROWS) return null;
      const ngNew = gridCount(keptFirstT, keptLastT);
      tg = growF(tg, ngNew);
      for (let i = ng; i < ngNew; i++) tg[i] = gridTime(keptFirstT, i);
      ng = ngNew;
      valid = growU8(valid, ng);
      let vFrom = ng;
      for (let j = 0; j < 9; j++) {
        const from = updateColumn(cols[j]);
        if (j < 3 || j >= 6) vFrom = Math.min(vFrom, from);
      }
      vFrom = Math.min(vFrom, validDone);
      for (let i = vFrom; i < ng; i++) {
        let ok = 1;
        for (const j of [0, 1, 2, 6, 7, 8]) if (!Number.isFinite(cols[j].u[i])) ok = 0;
        valid[i] = ok;
      }
      validDone = Math.min(cols[0].done, cols[1].done, cols[2].done, cols[6].done, cols[7].done, cols[8].done);
      return build(raw);
    },
    reset(): void {
      init();
      batchOnly = quantized;
    },
    get stats(): StreamBuilderStats {
      return { rows: m, grid: ng, ...counters, batchOnly };
    },
  };
}
