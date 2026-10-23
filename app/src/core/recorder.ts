/**
 * Accumulates raw sensor samples of one trial into the columnar TrialRecord['streams'] layout.
 * Times are stored relative to t0 (the source-clock time recording started). Values are rounded
 * per ROUNDING when building — in practice as they are added (build() only copies), so
 * addMotionRel() can hand the online segmenter exactly the values that will be stored, and an
 * offline re-run on a stored record reproduces the online segmentation.
 * Missing values are stored as null (JSON has no NaN): null, undefined, NaN and ±Infinity all
 * become null, and an output never contains a non-finite number.
 * ts columns: event.timeStamp minus the first event's raw timeStamp (ts_first_raw keeps the raw).
 * "First event" = the first sample of that stream with a finite timeStamp; earlier samples get
 * ts = null. interval_raw is likewise the first finite event.interval, unconverted.
 *
 * Column types the schema fixes: t columns and quaternion x/y/z/w are numbers only, so
 *  - a motion sample with a non-finite t keeps its slot (stream indices must stay aligned with the
 *    segmenter) and repeats the previous t (0 for the first) — the TrialController drops such
 *    samples before they get here;
 *  - orientation/quaternion samples, touches and cues without a finite time are skipped, and so is a
 *    quaternion with a non-finite component.
 * Orientation / quaternion streams are null when no sample was recorded.
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import { GRAVITY, ROUNDING } from './constants';
import type { RelMotionSample } from './segmentation';
import type {
  CueKind,
  CueRecord,
  Euler,
  MotionSample,
  OrientationSample,
  QuaternionSample,
  QuaternionStream,
  TouchRecord,
  TrialQuality,
  TrialRecord,
  Vec3,
} from './types';

export interface RateStats {
  n: number;
  hz: number | null; // (n-1) / (t_last - t_first) * 1000
  dt_median: number | null;
  dt_p95: number | null;
  dt_max: number | null;
  gaps_over_50ms: number;
}

/** quality.warnings entry added by computeQuality when the motion rate is below this (Hz). */
export const LOW_MOTION_HZ = 40;
/**
 * Possible sensor saturation ('possible_gyro_clipping' / 'possible_acc_clipping'). The Android CDD
 * only guarantees ±1000 deg/s gyroscopes and some phones run the accelerometer at ±4 g, while fast
 * trials can reach ~4 g. Same thresholds as analysis/motionlab/qa.py.
 */
export const CLIP_GYRO_DPS = 950;
export const CLIP_ACC_MS2 = 3.5 * GRAVITY;

const POW10 = [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8];
const TWO_POW_52 = 4503599627370496;
/** Touch coordinates are CSS px; 0.01 px is far below finger accuracy. */
const TOUCH_XY_DECIMALS = 2;

/**
 * Rounds to `decimals` places, halves away from zero (same convention as util.roundOrNull);
 * null for null/undefined/non-finite. Never returns NaN or -0.
 */
export function roundValue(v: number | null | undefined, decimals: number): number | null {
  if (typeof v !== 'number' || !Number.isFinite(v)) return null;
  const f = POW10[decimals] ?? 10 ** decimals;
  const scaled = Math.abs(v) * f;
  // Beyond 2^52 a double has no fractional digits left at this scale (and v·f could overflow).
  if (!(scaled < TWO_POW_52)) return v;
  const r = (Math.sign(v) * Math.round(scaled)) / f;
  return r === 0 ? 0 : r;
}

function finiteOrNull(v: number | null | undefined): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** numpy-style percentile (linear interpolation between closest ranks) of a SORTED array. */
function quantileSorted(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * Sampling statistics of a strictly increasing list of times (ms). dt_median / dt_p95 use linear
 * interpolation (numpy.percentile default); a gap is a dt strictly greater than 50 ms. Non-finite
 * times are ignored. Fewer than 2 times → all statistics null; a zero span → hz null.
 */
export function rateStats(times: number[]): RateStats {
  const ts = times.filter((t) => typeof t === 'number' && Number.isFinite(t));
  const n = ts.length;
  if (n < 2) return { n, hz: null, dt_median: null, dt_p95: null, dt_max: null, gaps_over_50ms: 0 };
  const dts: number[] = new Array(n - 1);
  let gaps = 0;
  let max = -Infinity;
  for (let i = 1; i < n; i++) {
    const d = ts[i] - ts[i - 1];
    dts[i - 1] = d;
    if (d > 50) gaps++;
    if (d > max) max = d;
  }
  const span = ts[n - 1] - ts[0];
  dts.sort((a, b) => a - b);
  return {
    n,
    hz: span > 0 ? ((n - 1) / span) * 1000 : null,
    dt_median: quantileSorted(dts, 0.5),
    dt_p95: quantileSorted(dts, 0.95),
    dt_max: max,
    gaps_over_50ms: gaps,
  };
}

type Col = (number | null)[];

export class StreamRecorder {
  private readonly t0: number;

  private readonly mT: number[] = [];
  private readonly mTs: Col = [];
  private readonly mAcc: [Col, Col, Col] = [[], [], []];
  private readonly mAccg: [Col, Col, Col] = [[], [], []];
  private readonly mRr: [Col, Col, Col] = [[], [], []]; // alpha, beta, gamma
  private mInterval: number | null = null;
  private mTsFirst: number | null = null;

  private readonly oT: number[] = [];
  private readonly oTs: Col = [];
  private readonly oAlpha: Col = [];
  private readonly oBeta: Col = [];
  private readonly oGamma: Col = [];
  private readonly oCompass: Col = [];
  private oAbsolute: boolean | null = null;
  private oTsFirst: number | null = null;

  private readonly qT: number[] = [];
  private readonly qX: number[] = [];
  private readonly qY: number[] = [];
  private readonly qZ: number[] = [];
  private readonly qW: number[] = [];
  private qSensor: QuaternionStream['sensor'] | null = null;
  private qFreq: number | null = null;

  private readonly touches: TouchRecord[] = [];
  private readonly cues: CueRecord[] = [];
  private maxT: number | null = null;

  /** @param t0 source-clock time that becomes t = 0 */
  constructor(t0: number) {
    this.t0 = Number.isFinite(t0) ? t0 : 0;
  }

  /** Source-clock time → stored relative time (ms since t0, rounded); null if not finite. */
  rel(t: number): number | null {
    return roundValue(t - this.t0, ROUNDING.time);
  }

  /** Latest stored relative time over all streams, touches and cues (null when nothing stored). */
  get lastTime(): number | null {
    return this.maxT;
  }

  private seen(t: number): void {
    if (this.maxT === null || t > this.maxT) this.maxT = t;
  }

  /** Appends a sample (time in source clock); returns its index in the motion stream. */
  addMotion(s: MotionSample): number {
    return this.addMotionRel(s).index;
  }

  /**
   * Appends a sample and returns it exactly as stored: relative, rounded t/ts and values, vectors
   * null unless all 3 components are present. This is what the online segmenter is fed.
   */
  addMotionRel(s: MotionSample): RelMotionSample {
    const index = this.mT.length;
    let t = this.rel(s.t);
    if (t === null) t = index > 0 ? this.mT[index - 1] : 0;
    this.mT.push(t);
    this.seen(t);

    const tsRaw = finiteOrNull(s.ts);
    if (this.mTsFirst === null && tsRaw !== null) this.mTsFirst = tsRaw;
    const ts = tsRaw !== null && this.mTsFirst !== null ? roundValue(tsRaw - this.mTsFirst, ROUNDING.time) : null;
    this.mTs.push(ts);
    const interval = finiteOrNull(s.interval);
    if (this.mInterval === null && interval !== null) this.mInterval = interval;

    const acc = push3(this.mAcc, s.acc, ROUNDING.acc);
    const accg = push3(this.mAccg, s.accg, ROUNDING.acc);
    const rr = s.rr;
    const ra = roundValue(rr?.alpha, ROUNDING.rr);
    const rb = roundValue(rr?.beta, ROUNDING.rr);
    const rg = roundValue(rr?.gamma, ROUNDING.rr);
    this.mRr[0].push(ra);
    this.mRr[1].push(rb);
    this.mRr[2].push(rg);
    const rrOut: Euler | null = ra !== null && rb !== null && rg !== null ? { alpha: ra, beta: rb, gamma: rg } : null;
    return { t, ts, acc, accg, rr: rrOut, interval, index };
  }

  addOrientation(s: OrientationSample): void {
    const t = this.rel(s.t);
    if (t === null) return;
    this.oT.push(t);
    this.seen(t);
    const tsRaw = finiteOrNull(s.ts);
    if (this.oTsFirst === null && tsRaw !== null) this.oTsFirst = tsRaw;
    this.oTs.push(tsRaw !== null && this.oTsFirst !== null ? roundValue(tsRaw - this.oTsFirst, ROUNDING.time) : null);
    this.oAlpha.push(roundValue(s.alpha, ROUNDING.angle));
    this.oBeta.push(roundValue(s.beta, ROUNDING.angle));
    this.oGamma.push(roundValue(s.gamma, ROUNDING.angle));
    this.oCompass.push(roundValue(s.compass, ROUNDING.angle));
    if (this.oAbsolute === null && typeof s.absolute === 'boolean') this.oAbsolute = s.absolute;
  }

  addQuaternion(s: QuaternionSample, frequencyHz: number): void {
    const t = this.rel(s.t);
    const q = s.q;
    if (t === null || !q) return;
    const x = roundValue(q[0], ROUNDING.quat);
    const y = roundValue(q[1], ROUNDING.quat);
    const z = roundValue(q[2], ROUNDING.quat);
    const w = roundValue(q[3], ROUNDING.quat);
    if (x === null || y === null || z === null || w === null) return;
    this.qT.push(t);
    this.qX.push(x);
    this.qY.push(y);
    this.qZ.push(z);
    this.qW.push(w);
    this.seen(t);
    if (this.qSensor === null) {
      this.qSensor = s.sensor === 'AbsoluteOrientationSensor' ? 'AbsoluteOrientationSensor' : 'RelativeOrientationSensor';
    }
    if (this.qFreq === null && Number.isFinite(frequencyHz)) this.qFreq = frequencyHz;
  }

  /** t in source clock */
  addTouch(t: number, type: 'start' | 'end' | 'cancel', n: number, x: number | null, y: number | null): void {
    const rt = this.rel(t);
    if (rt === null) return;
    this.touches.push({
      t: rt,
      type,
      n: Number.isFinite(n) && n > 0 ? Math.round(n) : 0,
      x: roundValue(x, TOUCH_XY_DECIMALS),
      y: roundValue(y, TOUCH_XY_DECIMALS),
    });
    this.seen(rt);
  }

  /** t in source clock */
  addCue(t: number, kind: CueKind): void {
    const rt = this.rel(t);
    if (rt === null) return;
    this.cues.push({ t: rt, kind });
    this.seen(rt);
  }

  get motionCount(): number {
    return this.mT.length;
  }

  /** Relative time (ms since t0) of motion sample i. */
  motionTime(i: number): number {
    if (!Number.isInteger(i) || i < 0 || i >= this.mT.length) {
      throw new RangeError(`motion sample ${i} does not exist (${this.mT.length} recorded)`);
    }
    return this.mT[i];
  }

  /** Builds the rounded, columnar streams object (arrays copied). */
  build(): TrialRecord['streams'] {
    return {
      motion: {
        t: this.mT.slice(),
        ts: this.mTs.slice(),
        acc_x: this.mAcc[0].slice(),
        acc_y: this.mAcc[1].slice(),
        acc_z: this.mAcc[2].slice(),
        accg_x: this.mAccg[0].slice(),
        accg_y: this.mAccg[1].slice(),
        accg_z: this.mAccg[2].slice(),
        rr_alpha: this.mRr[0].slice(),
        rr_beta: this.mRr[1].slice(),
        rr_gamma: this.mRr[2].slice(),
        interval_raw: this.mInterval,
        ts_first_raw: this.mTsFirst,
      },
      orientation:
        this.oT.length > 0
          ? {
              t: this.oT.slice(),
              ts: this.oTs.slice(),
              alpha: this.oAlpha.slice(),
              beta: this.oBeta.slice(),
              gamma: this.oGamma.slice(),
              compass: this.oCompass.slice(),
              absolute: this.oAbsolute,
              ts_first_raw: this.oTsFirst,
            }
          : null,
      quaternion:
        this.qT.length > 0
          ? {
              t: this.qT.slice(),
              x: this.qX.slice(),
              y: this.qY.slice(),
              z: this.qZ.slice(),
              w: this.qW.slice(),
              sensor: this.qSensor ?? 'RelativeOrientationSensor',
              frequency_hz: this.qFreq ?? 0,
            }
          : null,
      touches: this.touches.map((x) => ({ ...x })),
      cues: this.cues.map((c) => ({ ...c })),
    };
  }
}

/** Rounds and appends one vector to its 3 columns; returns it as stored (null unless complete). */
function push3(cols: [Col, Col, Col], v: Vec3 | null | undefined, decimals: number): Vec3 | null {
  const x = roundValue(v?.[0], decimals);
  const y = roundValue(v?.[1], decimals);
  const z = roundValue(v?.[2], decimals);
  cols[0].push(x);
  cols[1].push(y);
  cols[2].push(z);
  return x !== null && y !== null && z !== null ? [x, y, z] : null;
}

const r3 = (v: number | null): number | null => roundValue(v, 3);

/**
 * Quality summary for a built streams object. `warnings` are copied first; computeQuality then
 * appends (once each) what it can see in the streams: 'no_motion_samples', 'acc_unavailable' /
 * 'rotation_unavailable' (no sample with all 3 components), and 'low_motion_rate' (< LOW_MOTION_HZ).
 * Rates and dts are rounded to 0.001.
 */
export function computeQuality(
  streams: TrialRecord['streams'],
  hiddenDuring: boolean,
  warnings: string[],
): TrialQuality {
  const m = streams.motion;
  const n = m.t.length;
  const rs = rateStats(m.t);
  let accAvailable = false;
  let rotationAvailable = false;
  for (let i = 0; i < n && !(accAvailable && rotationAvailable); i++) {
    if (!accAvailable && isNum(m.acc_x[i]) && isNum(m.acc_y[i]) && isNum(m.acc_z[i])) accAvailable = true;
    if (!rotationAvailable && isNum(m.rr_alpha[i]) && isNum(m.rr_beta[i]) && isNum(m.rr_gamma[i])) {
      rotationAvailable = true;
    }
  }
  const out: string[] = [];
  const add = (w: string): void => {
    if (!out.includes(w)) out.push(w);
  };
  for (const w of warnings) add(w);
  if (n === 0) add('no_motion_samples');
  else {
    if (!accAvailable) add('acc_unavailable');
    if (!rotationAvailable) add('rotation_unavailable');
  }
  if (rs.hz !== null && rs.hz < LOW_MOTION_HZ) add('low_motion_rate');
  for (let i = 0; i < n; i++) {
    const a = m.rr_alpha[i];
    const b = m.rr_beta[i];
    const g = m.rr_gamma[i];
    if (isNum(a) && isNum(b) && isNum(g) && Math.hypot(a as number, b as number, g as number) > CLIP_GYRO_DPS) {
      add('possible_gyro_clipping');
    }
    const x = m.accg_x[i];
    const y = m.accg_y[i];
    const z = m.accg_z[i];
    if (isNum(x) && isNum(y) && isNum(z) && Math.hypot(x as number, y as number, z as number) > CLIP_ACC_MS2) {
      add('possible_acc_clipping');
    }
  }
  return {
    n_motion: n,
    n_orientation: streams.orientation?.t.length ?? 0,
    n_quaternion: streams.quaternion?.t.length ?? 0,
    motion_hz: r3(rs.hz),
    motion_dt_ms_median: r3(rs.dt_median),
    motion_dt_ms_max: r3(rs.dt_max),
    gaps_over_50ms: rs.gaps_over_50ms,
    acc_available: accAvailable,
    rotation_available: rotationAvailable,
    hidden_during: hiddenDuring === true,
    warnings: out,
  };
}

function isNum(v: number | null | undefined): boolean {
  return typeof v === 'number' && Number.isFinite(v);
}
