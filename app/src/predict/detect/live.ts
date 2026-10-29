/**
 * Helpers for the live engine (APP_SPEC 4.2, PORT_SPEC 2.15). The engine recomputes the standardised stream of its
 * buffer every tick; these functions tell it which bouts of that prefix are final and what the UI hints are.
 *
 * Freeze rule: a bout is final once it has ended, its look-ahead is in the buffer (iEnd + nLook - 1 <= n - 1) and the
 * stream extends settleMs (1 s) past its decision time; its row (window, features, P(class), every model's output)
 * is then computed on that tick's prefix and never again. After 1 s the zero-phase low-pass no longer changes any
 * sample the bout depends on (to about 1e-13), so the frozen values equal those of the complete recording.
 * Alternative with O(1) work per sample: feed only the settled samples (t <= t_last - settleMs) to a CausalDetector
 * (bouts.ts); it emits exactly the bouts this rule freezes.
 *
 * Cost per tick: the engine passes a LiveDetectCache, which keeps the activity signal and the median grid step of
 * the previous tick and recomputes only from the first row that changed (bit for bit the batch values), so a tick
 * no longer sorts and refilters the whole stream (that cost used to grow with the session).
 *
 * The detector and its thresholds are lane m4's (fitted for m4's classifier; lane prod has not run).
 */
import type { Bout, DetectorBundle, Rows3, Stream } from '../types';
import { MIN_ACTIVITY_N, activity } from './activity';
import { NOMINAL_STEP_MS, boutConstants, findBouts, isFrozen, medianStep } from './bouts';
import { fillGapsRows, norm3, npMedian, scaledState, sosfiltStep } from './numeric';

/** APP_SPEC 4.2 defaults */
export const SETTLE_MS = 1000;
export const HINT_WINDOW_MS = 500;

export interface LiveBouts {
  readonly activity: Float64Array;
  /** every bout of the prefix (time order) */
  readonly bouts: readonly Bout[];
  /** frozen[k]: bouts[k] is final (the freeze rule) */
  readonly frozen: readonly boolean[];
  readonly nLook: number;
  /** the newest activity is below lo_dps (EngineSnapshot.still) */
  readonly still: boolean;
  /** median(diff(t)) of the stream (NOMINAL_STEP_MS with fewer than 2 samples) */
  readonly dtMs: number;
}

/**
 * The bouts of a live prefix and which of them are final. With a `cache` (one per stream, the engine's), the
 * activity and the median step are updated from the previous tick instead of recomputed: same values bit for bit.
 * The returned activity is then a view the next call overwrites (read it within the tick).
 */
export function liveBouts(stream: Stream, detector: DetectorBundle, settleMs: number = SETTLE_MS, cache?: LiveDetectCache): LiveBouts {
  const t = stream.t;
  const n = t.length;
  const p = detector.segParams;
  const pre = cache ? cache.update(stream) : null;
  const act = pre ? pre.act : activity(t, stream.gyro, p, { sos: detector.activitySos, zi: detector.activityZi });
  const dt = n >= 2 ? (pre ? pre.dtMs : medianStep(t)) : NOMINAL_STEP_MS;
  const bouts = findBouts(t, act, p, detector.causalDelayMs, n >= 2 ? dt : undefined);
  const { nLook } = boutConstants(p, dt, detector.causalDelayMs);
  const tLast = n ? t[n - 1] : NaN;
  return {
    activity: act,
    bouts,
    frozen: bouts.map((b) => isFrozen(b, nLook, n, tLast, settleMs)),
    nLook,
    still: n > 0 && act[n - 1] < p.lo_dps,
    dtMs: dt,
  };
}

type F64 = Float64Array<ArrayBuffer>;

function growTo(a: F64, need: number): F64 {
  if (need <= a.length) return a;
  let cap = Math.max(256, a.length);
  while (cap < need) cap *= 2;
  const b = new Float64Array(cap);
  b.set(a);
  return b;
}

/**
 * activity() and medianStep() of a GROWING live stream, carried over from the previous tick (one cache per stream).
 *
 * Exact, not approximate: the rows before the first row whose time or gyro changed are the same numbers, so their
 * activity is reused; the recomputation restarts after the last row s before that whose three gyro values are
 * finite. Gap filling (np.interp per column) of every later row then reads only rows >= s, exactly as on the whole
 * stream, and the causal filter continues from its saved state after row s (the recursion of sosfilt). The median
 * of the grid steps is kept as counts of the (few) distinct steps. A restarted or shorter stream, a stream shorter
 * than MIN_ACTIVITY_N (left unfiltered by activity()) or no such row s fall back to the batch computation.
 */
export class LiveDetectCache {
  private n = 0;
  private t: F64 = new Float64Array(0);
  private g: F64 = new Float64Array(0);
  private act: F64 = new Float64Array(0);
  /** the filter state after each row (zl values per row) */
  private z: F64 = new Float64Array(0);
  private readonly zl: number;
  private steps = new Map<number, number>();
  private nSteps = 0;

  constructor(private readonly detector: DetectorBundle) {
    this.zl = detector.activitySos.length * 2;
  }

  reset(): void {
    this.n = 0;
    this.steps = new Map();
    this.nSteps = 0;
  }

  /** the activity (a view of length n) and the median grid step of `stream` */
  update(stream: Stream): { readonly act: Float64Array; readonly dtMs: number } {
    const t = stream.t;
    const gy = stream.gyro;
    const n = gy.n;
    const p = this.detector.segParams;
    if (!p.smooth_hz || n < MIN_ACTIVITY_N) {
      this.reset();
      return { act: activity(t, gy, p, { sos: this.detector.activitySos, zi: this.detector.activityZi }), dtMs: medianStep(t) };
    }
    const gd = gy.data;
    const m = this.n <= n ? this.n : 0;
    let i0 = 0;
    while (
      i0 < m &&
      Object.is(t[i0], this.t[i0]) &&
      Object.is(gd[3 * i0], this.g[3 * i0]) &&
      Object.is(gd[3 * i0 + 1], this.g[3 * i0 + 1]) &&
      Object.is(gd[3 * i0 + 2], this.g[3 * i0 + 2])
    )
      i0++;
    for (let i = i0; i < m; i++) {
      if (!Object.is(t[i], this.t[i])) {
        i0 = 0; // another grid: a new stream
        break;
      }
    }
    let s = i0 - 1;
    while (s >= 0 && !(Number.isFinite(gd[3 * s]) && Number.isFinite(gd[3 * s + 1]) && Number.isFinite(gd[3 * s + 2]))) s--;
    if (s < 0) this.full(t, gy);
    else this.from(s, t, gy);
    // the grid steps: the rows below i0 are unchanged, so only the new steps are counted
    if (i0 === 0) {
      this.steps = new Map();
      this.nSteps = 0;
    }
    for (let i = this.nSteps; i < n - 1; i++) {
      const d = t[i + 1] - t[i];
      this.steps.set(d, (this.steps.get(d) ?? 0) + 1);
    }
    this.nSteps = n - 1;
    // the inputs, for the next comparison
    this.t = growTo(this.t, n);
    this.g = growTo(this.g, 3 * n);
    for (let i = i0; i < n; i++) this.t[i] = t[i];
    this.g.set(gd.subarray(3 * i0, 3 * n), 3 * i0);
    this.n = n;
    return { act: this.act.subarray(0, n), dtMs: this.median() };
  }

  /** np.median of the counted steps, with npMedian's arithmetic */
  private median(): number {
    const total = this.nSteps;
    if (total <= 0) return NaN;
    const keys = Float64Array.from(this.steps.keys()).sort();
    if (keys.some((k) => Number.isNaN(k))) return NaN;
    const at = (r: number): number => {
      let c = 0;
      for (const k of keys) {
        c += this.steps.get(k) ?? 0;
        if (r < c) return k;
      }
      return keys[keys.length - 1];
    };
    const h = total >> 1;
    return total % 2 === 1 ? at(h) : (at(h - 1) + at(h)) / 2;
  }

  private ensure(n: number): void {
    this.act = growTo(this.act, n);
    this.z = growTo(this.z, this.zl * n);
  }

  /** the whole stream: activity()'s fill, norm and sosfilt from zi * s[0], keeping the state after every row */
  private full(t: ArrayLike<number>, gy: Rows3): void {
    const n = gy.n;
    this.ensure(n);
    const w = fillGapsRows(t, gy).data;
    let z: Float64Array | null = null;
    for (let i = 0; i < n; i++) {
      const v = norm3(w[3 * i], w[3 * i + 1], w[3 * i + 2]);
      const x = Number.isFinite(v) ? v : 0.0;
      if (z === null) z = scaledState(this.detector.activityZi, x);
      this.act[i] = sosfiltStep(this.detector.activitySos, x, z);
      this.z.set(z, this.zl * i);
    }
  }

  /** rows s + 1 .. n - 1 from the state after row s (row s has three finite gyro values) */
  private from(s: number, t: ArrayLike<number>, gy: Rows3): void {
    const n = gy.n;
    this.ensure(n);
    if (s >= n - 1) return;
    const len = n - s;
    const ts = new Float64Array(len);
    for (let i = 0; i < len; i++) ts[i] = t[s + i];
    const w = fillGapsRows(ts, { n: len, data: gy.data.subarray(3 * s, 3 * n) }).data;
    const z = this.z.slice(this.zl * s, this.zl * (s + 1));
    for (let k = 1; k < len; k++) {
      const v = norm3(w[3 * k], w[3 * k + 1], w[3 * k + 2]);
      const x = Number.isFinite(v) ? v : 0.0;
      this.act[s + k] = sosfiltStep(this.detector.activitySos, x, z);
      this.z.set(z, this.zl * (s + k));
    }
  }
}

/**
 * A bout of a later tick is the same movement as an already frozen one when their onsets lie within half a grid
 * step (PORT_SPEC 2.15): the onset of a frozen bout no longer moves.
 */
export function sameBout(a: Bout, b: Bout, stepMs: number = NOMINAL_STEP_MS): boolean {
  return Math.abs(a.onsetMs - b.onsetMs) < stepMs / 2;
}

/**
 * EngineSnapshot.screenAwayHint (UI only, never changes a result): the activity stayed below lo_dps for the last
 * windowMs and the median of f_z / |f| over that window is below 0 (screen facing down or away).
 */
export function screenAwayHint(stream: Stream, act: Float64Array, detector: DetectorBundle, windowMs: number = HINT_WINDOW_MS): boolean {
  const t = stream.t;
  const n = t.length;
  if (n === 0 || t[n - 1] - t[0] < windowMs) return false;
  const from = t[n - 1] - windowMs;
  const upz: number[] = [];
  const f = stream.f.data;
  for (let i = n - 1; i >= 0 && t[i] >= from; i--) {
    if (!(act[i] < detector.segParams.lo_dps)) return false;
    const nrm = norm3(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
    if (Number.isFinite(nrm) && nrm > 0) upz.push(f[3 * i + 2] / nrm);
  }
  return upz.length > 0 && npMedian(upz) < 0;
}
