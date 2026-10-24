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
 *  - orientation/quaternion/magnetometer samples, screen events, touches and cues without a finite
 *    time are skipped, and so is a quaternion with a non-finite component.
 * Orientation / quaternion / magnetometer streams are null when no sample was recorded.
 *
 * Schema 1.2 additions (the v2 app always writes them; every one is optional in the schema):
 *  - orientation.compass_accuracy (iOS webkitCompassAccuracy, one entry per sample, null elsewhere);
 *  - streams.orientation_abs ('deviceorientationabsolute' events, same layout as orientation),
 *    streams.quaternion_abs (AbsoluteOrientationSensor), streams.magnetometer (µT, 3 decimals) — each
 *    null without a sample — and streams.screen_events ([] without a change);
 *  - touch contact details: radius_x / radius_y (CSS px, 0.01), force (1e-4), rotation_angle (deg,
 *    1e-3), x_norm / y_norm (fraction of the viewport, clamped to [0, 1], 1e-4); null when unknown;
 *  - quality.n_quaternion_abs / n_orientation_abs.
 *
 * IMPLEMENTED BY: segmentation agent.
 */
import { GRAVITY, ROUNDING } from './constants';
import type { RelMotionSample } from './segmentation';
import type {
  CueKind,
  CueRecord,
  Euler,
  MagnetometerSample,
  MagnetometerStream,
  MotionSample,
  OrientationSample,
  OrientationStream,
  QuaternionSample,
  QuaternionStream,
  ScreenEventRecord,
  ScreenSample,
  TouchDetails,
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
/** Touch coordinates and contact radii are CSS px; 0.01 px is far below finger accuracy. */
const TOUCH_XY_DECIMALS = 2;
/** Touch.force is 0..1. */
const TOUCH_FORCE_DECIMALS = 4;
/** x_norm / y_norm: 1e-4 of the viewport (≈ 0.04 px on a phone). */
const TOUCH_NORM_DECIMALS = 4;
/** Magnetometer readings (µT): the earth's field is 25–65 µT, sensor noise ~0.5 µT. */
const MAGNETOMETER_DECIMALS = 3;

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

/** A normalised coordinate: rounded, clamped to [0, 1]; null when unknown. */
function unitOrNull(v: number | null | undefined): number | null {
  const r = roundValue(v, TOUCH_NORM_DECIMALS);
  return r === null ? null : Math.min(1, Math.max(0, r));
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

/** Columns of one deviceorientation-style stream (relative or absolute). */
class OrientationColumns {
  readonly t: number[] = [];
  readonly ts: Col = [];
  readonly alpha: Col = [];
  readonly beta: Col = [];
  readonly gamma: Col = [];
  readonly compass: Col = [];
  readonly compassAccuracy: Col = [];
  absolute: boolean | null = null;
  tsFirst: number | null = null;

  /** Appends a sample at relative time t (already rounded). */
  add(t: number, s: OrientationSample): void {
    this.t.push(t);
    const tsRaw = finiteOrNull(s.ts);
    if (this.tsFirst === null && tsRaw !== null) this.tsFirst = tsRaw;
    this.ts.push(tsRaw !== null && this.tsFirst !== null ? roundValue(tsRaw - this.tsFirst, ROUNDING.time) : null);
    this.alpha.push(roundValue(s.alpha, ROUNDING.angle));
    this.beta.push(roundValue(s.beta, ROUNDING.angle));
    this.gamma.push(roundValue(s.gamma, ROUNDING.angle));
    this.compass.push(roundValue(s.compass, ROUNDING.angle));
    this.compassAccuracy.push(roundValue(s.compass_accuracy, ROUNDING.angle));
    if (this.absolute === null && typeof s.absolute === 'boolean') this.absolute = s.absolute;
  }

  build(): OrientationStream | null {
    if (this.t.length === 0) return null;
    return {
      t: this.t.slice(),
      ts: this.ts.slice(),
      alpha: this.alpha.slice(),
      beta: this.beta.slice(),
      gamma: this.gamma.slice(),
      compass: this.compass.slice(),
      absolute: this.absolute,
      ts_first_raw: this.tsFirst,
      compass_accuracy: this.compassAccuracy.slice(),
    };
  }
}

/** Columns of one Generic Sensor quaternion stream. */
class QuaternionColumns {
  readonly t: number[] = [];
  readonly x: number[] = [];
  readonly y: number[] = [];
  readonly z: number[] = [];
  readonly w: number[] = [];
  sensor: QuaternionStream['sensor'] | null = null;
  freq: number | null = null;

  /** Appends a sample at relative time t; false (nothing stored) when a component is not finite. */
  add(t: number, s: QuaternionSample, frequencyHz: number): boolean {
    const q = s.q;
    if (!q) return false;
    const x = roundValue(q[0], ROUNDING.quat);
    const y = roundValue(q[1], ROUNDING.quat);
    const z = roundValue(q[2], ROUNDING.quat);
    const w = roundValue(q[3], ROUNDING.quat);
    if (x === null || y === null || z === null || w === null) return false;
    this.t.push(t);
    this.x.push(x);
    this.y.push(y);
    this.z.push(z);
    this.w.push(w);
    if (this.sensor === null) {
      this.sensor = s.sensor === 'AbsoluteOrientationSensor' ? 'AbsoluteOrientationSensor' : 'RelativeOrientationSensor';
    }
    if (this.freq === null && Number.isFinite(frequencyHz)) this.freq = frequencyHz;
    return true;
  }

  build(fallbackSensor: QuaternionStream['sensor']): QuaternionStream | null {
    if (this.t.length === 0) return null;
    return {
      t: this.t.slice(),
      x: this.x.slice(),
      y: this.y.slice(),
      z: this.z.slice(),
      w: this.w.slice(),
      sensor: this.sensor ?? fallbackSensor,
      frequency_hz: this.freq ?? 0,
    };
  }
}

export class StreamRecorder {
  private readonly t0: number;

  private readonly mT: number[] = [];
  private readonly mTs: Col = [];
  private readonly mAcc: [Col, Col, Col] = [[], [], []];
  private readonly mAccg: [Col, Col, Col] = [[], [], []];
  private readonly mRr: [Col, Col, Col] = [[], [], []]; // alpha, beta, gamma
  private mInterval: number | null = null;
  private mTsFirst: number | null = null;

  private readonly orientation = new OrientationColumns();
  private readonly orientationAbs = new OrientationColumns();
  private readonly quaternion = new QuaternionColumns();
  private readonly quaternionAbs = new QuaternionColumns();

  private readonly magT: number[] = [];
  private readonly magXyz: [Col, Col, Col] = [[], [], []];
  private magFreq: number | null = null;

  private readonly screens: ScreenEventRecord[] = [];
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

  /** 'deviceorientation' (relative on Chrome Android, absolute where the browser says so). */
  addOrientation(s: OrientationSample): void {
    const t = this.rel(s.t);
    if (t === null) return;
    this.orientation.add(t, s);
    this.seen(t);
  }

  /** schema 1.2: 'deviceorientationabsolute' → streams.orientation_abs. */
  addOrientationAbs(s: OrientationSample): void {
    const t = this.rel(s.t);
    if (t === null) return;
    this.orientationAbs.add(t, s);
    this.seen(t);
  }

  /** RelativeOrientationSensor (or any quaternion sensor) → streams.quaternion. */
  addQuaternion(s: QuaternionSample, frequencyHz: number): void {
    const t = this.rel(s.t);
    if (t === null || !s.q) return;
    if (this.quaternion.add(t, s, frequencyHz)) this.seen(t);
  }

  /** schema 1.2: AbsoluteOrientationSensor → streams.quaternion_abs (sensor always 'AbsoluteOrientationSensor'). */
  addQuaternionAbs(s: QuaternionSample, frequencyHz: number): void {
    const t = this.rel(s.t);
    if (t === null || !s.q) return;
    if (this.quaternionAbs.add(t, { ...s, sensor: 'AbsoluteOrientationSensor' }, frequencyHz)) this.seen(t);
  }

  /** schema 1.2: Magnetometer readings (µT) → streams.magnetometer. */
  addMagnetometer(s: MagnetometerSample, frequencyHz: number): void {
    const t = this.rel(s.t);
    if (t === null) return;
    this.magT.push(t);
    this.magXyz[0].push(roundValue(s.x, MAGNETOMETER_DECIMALS));
    this.magXyz[1].push(roundValue(s.y, MAGNETOMETER_DECIMALS));
    this.magXyz[2].push(roundValue(s.z, MAGNETOMETER_DECIMALS));
    if (this.magFreq === null && Number.isFinite(frequencyHz)) this.magFreq = frequencyHz;
    this.seen(t);
  }

  /** schema 1.2: a screen orientation change → streams.screen_events. */
  addScreen(s: ScreenSample): void {
    const t = this.rel(s.t);
    if (t === null) return;
    this.screens.push({
      t,
      type: typeof s.type === 'string' ? s.type : null,
      angle: roundValue(s.angle, ROUNDING.angle),
    });
    this.seen(t);
  }

  /** t in source clock; `details` = the contact of the same touch as x / y (schema 1.2, null when absent). */
  addTouch(
    t: number,
    type: 'start' | 'end' | 'cancel',
    n: number,
    x: number | null,
    y: number | null,
    details?: Partial<TouchDetails> | null,
  ): void {
    const rt = this.rel(t);
    if (rt === null) return;
    const d = details ?? {};
    this.touches.push({
      t: rt,
      type,
      n: Number.isFinite(n) && n > 0 ? Math.round(n) : 0,
      x: roundValue(x, TOUCH_XY_DECIMALS),
      y: roundValue(y, TOUCH_XY_DECIMALS),
      radius_x: roundValue(d.radius_x, TOUCH_XY_DECIMALS),
      radius_y: roundValue(d.radius_y, TOUCH_XY_DECIMALS),
      force: roundValue(d.force, TOUCH_FORCE_DECIMALS),
      rotation_angle: roundValue(d.rotation_angle, ROUNDING.angle),
      x_norm: unitOrNull(d.x_norm),
      y_norm: unitOrNull(d.y_norm),
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
    const magnetometer: MagnetometerStream | null =
      this.magT.length > 0
        ? {
            t: this.magT.slice(),
            x: this.magXyz[0].slice(),
            y: this.magXyz[1].slice(),
            z: this.magXyz[2].slice(),
            frequency_hz: this.magFreq ?? 0,
          }
        : null;
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
      orientation: this.orientation.build(),
      quaternion: this.quaternion.build('RelativeOrientationSensor'),
      touches: this.touches.map((x) => ({ ...x })),
      cues: this.cues.map((c) => ({ ...c })),
      quaternion_abs: this.quaternionAbs.build('AbsoluteOrientationSensor'),
      orientation_abs: this.orientationAbs.build(),
      screen_events: this.screens.map((s) => ({ ...s })),
      magnetometer,
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
 * Rates and dts are rounded to 0.001. n_quaternion_abs / n_orientation_abs (schema 1.2) are always set.
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
    n_quaternion_abs: streams.quaternion_abs?.t.length ?? 0,
    n_orientation_abs: streams.orientation_abs?.t.length ?? 0,
  };
}

function isNum(v: number | null | undefined): boolean {
  return typeof v === 'number' && Number.isFinite(v);
}
