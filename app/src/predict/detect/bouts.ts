/**
 * Hysteresis bouts of the activity signal and their look-ahead confirmation (m4_detect.find_segments,
 * analysis/stage2/lanes/m4_detect.py:147-203).
 *
 *  - findBouts(): the batch function, an exact port (the reference the parity tests compare);
 *  - BoutTracker: the same bouts from activity samples fed one at a time. A bout is emitted as soon as its
 *    look-ahead is complete (sample iEnd + nLook - 1 has arrived); finish() emits what the batch function gives at
 *    the end of a stream (look-ahead cut short, the unended last bout). Fed a whole stream, it returns exactly
 *    findBouts() (same dt), checked by the tests;
 *  - CausalDetector: ActivityFilter + BoutTracker, fed grid rows of the gyro.
 *
 * Live use (APP_SPEC 4.2, PORT_SPEC 2.15): the standardised stream is zero-phase low-passed, so its newest samples
 * still change when later samples arrive. Feed the detector only the SETTLED part of the stream (t <= t_last -
 * settleMs): a bout is then emitted exactly when decided_ms + settleMs <= t_last, which is the freeze rule, and
 * nothing it read changes afterwards (to about 1e-13).
 *
 * The parameters (lo 15 / hi 30 deg/s, end_sustain 250 ms, ...) are lane m4's fitted SEG_GRID entry from the exported
 * bundle; they were fitted for lane m4's own classifier (lane prod, which would refit them for the final CNN, has not
 * run).
 */
import type { Bout, SegParams } from '../types';
import { npMax, npMedian, pyMin, roundHalfEven } from './numeric';
import { ActivityFilter, type ActivityFilterCoefs } from './activity';

/** The sample counts find_segments derives from the grid step (m4_detect.py:159-161). */
export interface BoutConstants {
  /** grid step used for the counts and the stillness durations (ms): median(diff(t)) in find_segments */
  readonly dtMs: number;
  /** causal filter delay subtracted from the bounds (ms) */
  readonly lagMs: number;
  readonly nEnd: number;
  readonly nMove: number;
  readonly nLook: number;
}

/** np.median(np.diff(t)) (m4_detect.py:155); NaN with fewer than 2 samples. */
export function medianStep(t: ArrayLike<number>): number {
  const n = t.length;
  if (n < 2) return NaN;
  const d = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) d[i] = t[i + 1] - t[i];
  return npMedian(d);
}

export function boutConstants(p: SegParams, dtMs: number, lagMs: number): BoutConstants {
  const nEnd = Math.max(1, roundHalfEven(p.end_sustain_ms / dtMs));
  const nMove = Math.max(1, roundHalfEven(p.sustain_ms / dtMs));
  const nLook = Math.max(nEnd, roundHalfEven((p.lookahead_s * 1000.0) / dtMs));
  return { dtMs, lagMs, nEnd, nMove, nLook };
}

/** The grid step of production streams (1000 / 60 ms); find_segments uses the median step of the stream itself. */
export const NOMINAL_STEP_MS = 1000.0 / 60.0;

/** A bout's raw bounds before the filters of find_segments: activity indices [a, b) and its previous separator. */
interface RawBout {
  readonly a: number;
  readonly b: number;
  readonly ks: number | null;
}

/** true when act[a..b-1] holds a run of at least nMove samples above hi_dps (m4_detect.py:177-179). */
function hasMovement(act: ArrayLike<number>, a: number, b: number, hi: number, nMove: number): boolean {
  let run = 0;
  for (let j = a; j < b; j++) {
    if (act[j] > hi) {
      run++;
      if (run >= nMove) return true;
    } else run = 0;
  }
  return false;
}

/**
 * The part of a bout known when it closes (m4_detect.py:176-190): null when find_segments drops it (no sustained
 * movement, or shorter than min_duration_ms). `n` is the stream length (only read for an unended bout).
 */
function boutHead(
  t: ArrayLike<number>,
  act: ArrayLike<number>,
  raw: RawBout,
  n: number,
  seps: readonly (readonly [number, number])[],
  p: SegParams,
  c: BoutConstants,
): Omit<Bout, 'index' | 'decidedMs' | 'stillAfterS'> | null {
  const { a, b, ks } = raw;
  if (!hasMovement(act, a, b, p.hi_dps, c.nMove)) return null;
  const ended = b < n;
  const iOn = a;
  const iEnd = ended ? b : n - 1; // the end is the FIRST still sample of the closing separator
  const onsetMs = t[iOn] - c.lagMs;
  const endMs = t[iEnd] - c.lagMs;
  if (endMs - onsetMs < p.min_duration_ms) return null; // max_duration_s is not applied (m4_detect.py:184)
  let sb: number;
  let startedActive: boolean;
  if (ks === null) {
    sb = 0.0;
    startedActive = true;
  } else {
    const [sa, sbEnd] = seps[ks];
    sb = (t[sbEnd - 1] - t[sa] + c.dtMs) / 1000.0; // m4_detect.py:190
    startedActive = false;
  }
  return {
    onsetMs,
    endMs,
    iOn,
    iEnd,
    stillBeforeS: pyMin(sb, p.still_cap_s),
    startedActive,
    ended,
    peakDps: npMax(act, iOn, iEnd - iOn + 1), // max(act[i_on .. i_end]) inclusive (m4_detect.py:202)
  };
}

/** The look-ahead part (m4_detect.py:192-199), evaluated on the first n samples. */
function boutTail(
  t: ArrayLike<number>,
  act: ArrayLike<number>,
  iEnd: number,
  ended: boolean,
  n: number,
  p: SegParams,
  c: BoutConstants,
): { stillAfterS: number; decidedMs: number } {
  if (!ended) return { stillAfterS: 0.0, decidedMs: t[n - 1] };
  let j = iEnd;
  while (j < n && act[j] < p.lo_dps && j - iEnd < c.nLook) j++;
  const sa = ((j - iEnd) * c.dtMs) / 1000.0;
  const decI = Math.min(n - 1, iEnd + c.nLook - 1);
  return { stillAfterS: pyMin(sa, p.lookahead_s), decidedMs: t[decI] };
}

/**
 * m4_detect.find_segments(t, gyro, p, s=act): every candidate bout in time order. `lagMs` is
 * m4_detect.causal_delay_ms(p) from the bundle (DetectorBundle.causalDelayMs). dt = median(diff(t)); a caller that
 * already has that exact value (the live engine's LiveDetectCache) passes it as `dtMs` to skip the sort.
 */
export function findBouts(t: ArrayLike<number>, act: ArrayLike<number>, p: SegParams, lagMs: number, dtMs?: number): Bout[] {
  const n = t.length;
  if (n < 3) return [];
  const c = boutConstants(p, dtMs ?? medianStep(t), lagMs);
  // separators: still runs of at least nEnd samples, or touching the stream start (m4_detect.py:162-164)
  const seps: [number, number][] = [];
  for (let i = 0; i < n; ) {
    if (act[i] < p.lo_dps) {
      let j = i;
      while (j < n && act[j] < p.lo_dps) j++;
      if (j - i >= c.nEnd || i === 0) seps.push([i, j]);
      i = j;
    } else i++;
  }
  const raws: RawBout[] = [];
  let prevEnd = 0;
  let prevSep: number | null = null;
  seps.forEach(([a, b], k) => {
    if (a > prevEnd) raws.push({ a: prevEnd, b: a, ks: prevSep });
    prevEnd = b;
    prevSep = k;
  });
  if (prevEnd < n) raws.push({ a: prevEnd, b: n, ks: prevSep });
  const out: Bout[] = [];
  for (const raw of raws) {
    const head = boutHead(t, act, raw, n, seps, p, c);
    if (head === null) continue;
    out.push({ index: out.length, ...head, ...boutTail(t, act, head.iEnd, head.ended, n, p, c) });
  }
  return out;
}

/**
 * The live freeze rule (PORT_SPEC 2.15, APP_SPEC 4.2): a bout of a prefix of n grid samples (last time tLast) is
 * final once it has ended, its whole look-ahead is in the buffer and the stream extends settleMs past its decision.
 */
export function isFrozen(b: Bout, nLook: number, n: number, tLast: number, settleMs: number): boolean {
  return b.ended && b.iEnd + nLook - 1 <= n - 1 && b.decidedMs + settleMs <= tLast;
}

/**
 * find_segments fed one activity sample at a time (module docstring). `dtMs` replaces find_segments' median step
 * (pass medianStep(t) of the whole stream for bitwise parity, or NOMINAL_STEP_MS live).
 */
export class BoutTracker {
  readonly constants: BoutConstants;
  private readonly t: number[] = [];
  private readonly act: number[] = [];
  /** separators so far; an ongoing one has b = -1 */
  private readonly seps: [number, number][] = [];
  private stillStart = -1;
  private sepOngoing = -1;
  private prevEnd = 0;
  private prevSep: number | null = null;
  /** closed, kept bouts waiting for their look-ahead */
  private readonly waiting: Omit<Bout, 'decidedMs' | 'stillAfterS'>[] = [];
  private kept = 0;
  private done = false;

  constructor(
    private readonly p: SegParams,
    lagMs: number,
    dtMs: number = NOMINAL_STEP_MS,
  ) {
    this.constants = boutConstants(p, dtMs, lagMs);
  }

  /** samples received */
  get n(): number {
    return this.t.length;
  }

  /** the newest activity value is below lo_dps (NaN-free; false before the first sample) */
  get still(): boolean {
    return this.stillStart >= 0;
  }

  /** the newest activity value (NaN before the first sample) */
  get lastActivity(): number {
    return this.act.length ? this.act[this.act.length - 1] : NaN;
  }

  /** onset of the bout in progress (ms, lag-corrected) when one is open and not yet closed; null otherwise */
  get openOnsetMs(): number | null {
    if (this.sepOngoing >= 0 || this.t.length === 0 || this.prevEnd >= this.t.length) return null;
    return this.t[this.prevEnd] - this.constants.lagMs;
  }

  /** Adds the activity sample (t ms, act deg/s) and returns the bouts whose look-ahead it completes. */
  push(t: number, act: number): Bout[] {
    if (this.done) throw new Error('BoutTracker: push after finish');
    const i = this.t.length;
    this.t.push(t);
    this.act.push(act);
    const c = this.constants;
    if (act < this.p.lo_dps) {
      if (this.stillStart < 0) this.stillStart = i;
      if (this.sepOngoing < 0 && (i - this.stillStart + 1 >= c.nEnd || this.stillStart === 0)) {
        // the still run becomes a separator: the bout before it ends at its first sample
        this.seps.push([this.stillStart, -1]);
        this.sepOngoing = this.seps.length - 1;
        if (this.stillStart > this.prevEnd) this.close({ a: this.prevEnd, b: this.stillStart, ks: this.prevSep });
      }
    } else if (this.stillStart >= 0) {
      if (this.sepOngoing >= 0) {
        this.seps[this.sepOngoing][1] = i;
        this.prevEnd = i;
        this.prevSep = this.sepOngoing;
      }
      this.stillStart = -1;
      this.sepOngoing = -1;
    }
    return this.emit(false);
  }

  /** End of the stream: the bouts the batch function adds at the end (cut look-ahead, the unended last bout). */
  finish(): Bout[] {
    if (this.done) return [];
    this.done = true;
    const n = this.t.length;
    if (n < 3) {
      this.waiting.length = 0;
      return [];
    }
    if (this.sepOngoing >= 0) {
      this.seps[this.sepOngoing][1] = n; // the separator runs to the end: no trailing bout
      this.prevEnd = n;
      this.prevSep = this.sepOngoing;
    }
    if (this.prevEnd < n) this.close({ a: this.prevEnd, b: n, ks: this.prevSep }); // possibly not ended
    return this.emit(true);
  }

  private close(raw: RawBout): void {
    const head = boutHead(this.t, this.act, raw, this.t.length, this.seps, this.p, this.constants);
    if (head === null) return;
    this.waiting.push({ index: this.kept++, ...head });
  }

  private emit(atEnd: boolean): Bout[] {
    const n = this.t.length;
    const out: Bout[] = [];
    while (this.waiting.length) {
      const h = this.waiting[0];
      if (!atEnd && !(h.ended && h.iEnd + this.constants.nLook - 1 <= n - 1)) break;
      this.waiting.shift();
      out.push({ ...h, ...boutTail(this.t, this.act, h.iEnd, h.ended, n, this.p, this.constants) });
    }
    return out;
  }
}

/** One gyro grid row in, the bouts it confirms out: ActivityFilter followed by a BoutTracker. */
export class CausalDetector {
  readonly filter: ActivityFilter;
  readonly tracker: BoutTracker;

  constructor(p: SegParams, coefs: ActivityFilterCoefs, lagMs: number, dtMs: number = NOMINAL_STEP_MS) {
    this.filter = new ActivityFilter(p, coefs);
    this.tracker = new BoutTracker(p, lagMs, dtMs);
  }

  push(t: number, gx: number, gy: number, gz: number): Bout[] {
    const out: Bout[] = [];
    for (const s of this.filter.push(t, gx, gy, gz)) out.push(...this.tracker.push(s.t, s.act));
    return out;
  }

  finish(): Bout[] {
    const out: Bout[] = [];
    for (const s of this.filter.finish()) out.push(...this.tracker.push(s.t, s.act));
    out.push(...this.tracker.finish());
    return out;
  }
}
