/**
 * The three logistic hand models. Each reads the same float32 (17, 64) approach window, as float64.
 *
 *   m4     lane m4's production hand model (m4_cal_nested): Yaw8 (analysis/stage2/lanes/m4_models.py:57-77),
 *          x = nan_to_num(window[yaw_cum][0::8]); z_raw = coef.x + intercept; P = the nested calibrator (the
 *          all-persons fit chose isotonic_sym); z = m4_models.logit(P) (eps 1e-6); the attempt's own displayed P is
 *          1 / (1 + exp(-z)) (m4_production.py:114). Always available.
 *   yaw    the reference (example_yaw_logreg, analysis/stage2/scripts/reference.py:64-85; analysis/stage3/runtime.py
 *          YawRef): x = window[yaw_cum][0::8] WITHOUT nan_to_num (a non-finite x makes it unavailable);
 *          z_raw = coef.x + intercept with the float32-valued coefficients in float64; P = expit(z_raw), uncalibrated;
 *          z = z_raw.
 *   phys6  lane m1's m1_lr_phys6_nested (runtime.Phys6, m1_features.compute_features, m1_models.Prep): six odd
 *          physics features (yaw, roll and lateral velocity at 50 % and 100 % of the approach), winsorised (asinh),
 *          scaled (rms), an antisymmetric logistic without intercept; P = clip(sigmoid(z_raw), 0, 1);
 *          z = m1_models.logit(P) (eps 1e-6).
 *
 * The dot products are summed left to right (scikit-learn's decision_function on one row matches that order,
 * checked against numpy); exp, log and asinh may differ from the C library by an ulp.
 */
import type { ApproachWindow, ModelAttemptOutput } from '../types';
import { makeCalibrator, type Calibrate } from './calibrators';
import type { M4ProductionJson, Phys6Json, YawRefJson } from './jsonTypes';
import { CHANNELS, N_NORM, channelIndex, isFiniteAll, logitRatio, nanToNum, npClip } from './numeric';

/** z = coef . x + intercept, summed left to right */
export function linearLogit(x: ArrayLike<number>, coef: ArrayLike<number>, intercept: number): number {
  let acc = 0.0;
  for (let i = 0; i < coef.length; i++) acc += x[i] * coef[i];
  return acc + intercept;
}

function checkWindow(w: ApproachWindow): Float32Array {
  const d = w.data;
  if (d.length !== CHANNELS.length * N_NORM) {
    throw new RangeError(`window has ${d.length} values, expected ${CHANNELS.length * N_NORM}`);
  }
  return d;
}

/** window[row][0::step], n values, as float64 */
export function sampleRow(data: ArrayLike<number>, row: number, step: number, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = data[row * N_NORM + i * step];
  return x;
}

const NAN_OUT = (id: 'cnn' | 'm4' | 'phys6' | 'yaw'): ModelAttemptOutput => ({
  modelId: id,
  zRaw: NaN,
  pRight: NaN,
  z: NaN,
  available: false,
});

// ───────────────────────────── lane m4: yaw8 + nested calibrator ─────────────────────────────

export interface M4Hand {
  readonly row: number;
  readonly step: number;
  readonly n: number;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nanToNum: boolean;
  readonly calibrate: Calibrate;
  readonly logitEps: number;
}

export function buildM4Hand(doc: M4ProductionJson): M4Hand {
  const h = doc.hand;
  if (h.kind !== 'yaw8') throw new RangeError(`m4_production.json: hand kind ${JSON.stringify(h.kind)}`);
  const row = channelIndex(h.channel);
  if (row !== h.index) throw new RangeError('m4_production.json: hand index does not match its channel');
  if (h.coef.length !== h.n || (h.n - 1) * h.step >= N_NORM) throw new RangeError('m4_production.json: hand shape');
  return {
    row,
    step: h.step,
    n: h.n,
    coef: Float64Array.from(h.coef),
    intercept: h.intercept,
    nanToNum: h.nanToNum,
    calibrate: makeCalibrator(doc.calibrator),
    logitEps: doc.logitEps,
  };
}

export interface M4Details {
  readonly output: ModelAttemptOutput;
  readonly x8: Float64Array;
  /** 1 / (1 + exp(-z)): the per-attempt P(right) predict_stream lists (m4_production.py:114, no clip) */
  readonly pAttempt: number;
}

export function m4Forward(m: M4Hand, window: ApproachWindow): M4Details {
  const data = checkWindow(window);
  const x8 = sampleRow(data, m.row, m.step, m.n);
  if (m.nanToNum) for (let i = 0; i < x8.length; i++) x8[i] = nanToNum(x8[i]);
  const zRaw = linearLogit(x8, m.coef, m.intercept);
  const p = m.calibrate(zRaw);
  const z = logitRatio(p, m.logitEps);
  return { output: { modelId: 'm4', zRaw, pRight: p, z, available: true }, x8, pAttempt: 1.0 / (1.0 + Math.exp(-z)) };
}

// ───────────────────────────── the yaw reference ─────────────────────────────

export interface YawRef {
  readonly row: number;
  readonly step: number;
  readonly n: number;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly calibrate: Calibrate;
}

export function buildYawRef(doc: YawRefJson): YawRef {
  if (doc.format !== 'nosetouch-yaw-ref-1') throw new RangeError(`yaw_ref.json: format ${JSON.stringify(doc.format)}`);
  const row = channelIndex(doc.channel);
  if (row !== doc.index) throw new RangeError('yaw_ref.json: index does not match its channel');
  if (doc.coef.length !== doc.n || (doc.n - 1) * doc.step >= N_NORM) throw new RangeError('yaw_ref.json: shape');
  if (doc.nan_to_num) throw new RangeError('yaw_ref.json: nan_to_num is not part of the reference');
  return {
    row,
    step: doc.step,
    n: doc.n,
    coef: Float64Array.from(doc.coef),
    intercept: doc.intercept,
    calibrate: makeCalibrator(doc.calibrator),
  };
}

export interface YawDetails {
  readonly output: ModelAttemptOutput;
  readonly x8: Float64Array;
}

export function yawForward(m: YawRef, window: ApproachWindow): YawDetails {
  const x8 = sampleRow(checkWindow(window), m.row, m.step, m.n);
  if (!isFiniteAll(x8)) return { output: NAN_OUT('yaw'), x8 };
  const zRaw = linearLogit(x8, m.coef, m.intercept);
  // predict_proba of a binary LogisticRegression = expit(decision_function) (scikit-learn _predict_proba_lr)
  const p = m.calibrate(zRaw);
  return { output: { modelId: 'yaw', zRaw, pRight: p, z: zRaw, available: true }, x8 };
}

// ───────────────────────────── PHYS6 (lane m1) ─────────────────────────────

/** The six features the export names, computed by phys6Features (m1_experiments.PHYS6). */
export const PHYS6_FEATURES = ['yaw_q50', 'yaw_q100', 'roll_q50', 'roll_q100', 'latvel_q50', 'latvel_q100'] as const;
type Phys6Name = (typeof PHYS6_FEATURES)[number];

export interface Phys6 {
  /** for each model column, its index in PHYS6_FEATURES */
  readonly cols: Int32Array;
  readonly winsor: 'none' | 'clip' | 'asinh';
  readonly med: Float64Array;
  readonly sw: Float64Array;
  readonly lo: Float64Array;
  readonly hi: Float64Array;
  readonly centre: Float64Array;
  readonly scale: Float64Array;
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly hMin: number;
  readonly calibrate: Calibrate;
  readonly logitEps: number;
}

export function buildPhys6(doc: Phys6Json): Phys6 {
  const bad = (msg: string): never => {
    throw new RangeError(`phys6.json: ${msg}`);
  };
  if (doc.format !== 'nosetouch-phys6-1') bad(`format ${JSON.stringify(doc.format)}`);
  const n = doc.features.length;
  // an 'ampnorm' choice would name *_an columns that need four more even features (PORT_SPEC 4.4): refuse it
  const cols = Int32Array.from(
    doc.features.map((f) => {
      const i = (PHYS6_FEATURES as readonly string[]).indexOf(f);
      if (i < 0) bad(`feature ${JSON.stringify(f)} is not computed by this port`);
      return i;
    }),
  );
  if (!['none', 'clip', 'asinh'].includes(doc.winsor)) bad(`winsor ${JSON.stringify(doc.winsor)}`);
  for (const [name, v] of [
    ['winsor_med', doc.winsor_med],
    ['winsor_s', doc.winsor_s],
    ['winsor_lo', doc.winsor_lo],
    ['winsor_hi', doc.winsor_hi],
    ['centre', doc.centre],
    ['scale', doc.scale],
    ['coef', doc.coef],
  ] as const) {
    if (v.length !== n) bad(`${name} has ${v.length} values, expected ${n}`);
  }
  if (!doc.input.body_frame.nan_below_h_min) bad('the m1 body frame is NaN below h_min');
  return {
    cols,
    winsor: doc.winsor,
    med: Float64Array.from(doc.winsor_med),
    sw: Float64Array.from(doc.winsor_s),
    lo: Float64Array.from(doc.winsor_lo.map((v) => (v === null ? -Infinity : v))),
    hi: Float64Array.from(doc.winsor_hi.map((v) => (v === null ? Infinity : v))),
    centre: Float64Array.from(doc.centre),
    scale: Float64Array.from(doc.scale),
    coef: Float64Array.from(doc.coef),
    intercept: doc.fit_intercept ? doc.intercept : 0.0,
    hMin: doc.input.body_frame.h_min,
    calibrate: makeCalibrator(doc.calibrator),
    logitEps: doc.logitEps,
  };
}

/** m1_features._cumtrapz (m1_features.py:246-250) of one row: c[0] = 0, c[k + 1] = c[k] + (0.5 * (x[k + 1] + x[k])) * dt */
function cumtrapz(x: Float64Array, dt: number): Float64Array {
  const T = x.length;
  const c = new Float64Array(T);
  let run = 0.0;
  for (let k = 0; k < T - 1; k++) {
    const inc = 0.5 * (x[k + 1] + x[k]) * dt;
    run = k === 0 ? inc : run + inc; // np.cumsum starts from the first increment itself
    c[k + 1] = run;
  }
  return c;
}

/** m1_features._at (m1_features.py:253-260): linear interpolation at the fraction q of the window */
function at(x: ArrayLike<number>, off: number, T: number, q: number): number {
  const pos = q * (T - 1);
  const i0 = Math.floor(pos);
  const i1 = Math.min(i0 + 1, T - 1);
  const f = pos - i0;
  return (1 - f) * x[off + i0] + f * x[off + i1];
}

export interface Phys6Features {
  /** PHYS6_FEATURES order; null when compute_features would refuse the trial */
  readonly values: Readonly<Record<Phys6Name, number>> | null;
  readonly durationS: number;
}

/**
 * The PHYS6 features of one window and its detected bounds (runtime.phys6_frame: m1_features.compute_features on
 * the window and its own duration). The body frame here is lane m1's (m1_features.py:229-243): float64, NOT rounded
 * to float32, and NaN where |e_z - (e_z.u) u| <= 1e-6 (the screen exactly horizontal), so the integrals become NaN.
 *
 * compute_features raises (and the model is unavailable) when the duration is not positive and finite or when ANY
 * catalogue feature is not finite; the catalogue reads every window channel (whole-channel NaN is the only NaN
 * pattern the pipeline produces), so this port requires a finite window and six finite features.
 */
export function phys6Features(window: ApproachWindow, hMin: number): Phys6Features {
  const A = checkWindow(window);
  const T = N_NORM;
  const durationS = (window.endMs - window.onsetMs) / 1000.0;
  if (!Number.isFinite(durationS) || durationS <= 0 || !isFiniteAll(A)) return { values: null, durationS };
  const dt = durationS / (T - 1);
  const iax = 0;
  const iwx = 3 * T;
  const iux = 6 * T;
  const roll = new Float64Array(T);
  const lat = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const ux = A[iux + t];
    const uy = A[iux + T + t];
    const uz = A[iux + 2 * T + t];
    let hx = -uz * ux;
    let hy = -uz * uy;
    let hz = -uz * uz;
    hz += 1.0;
    // np.linalg.norm(h, axis=1) = sqrt((h0 h0 + h1 h1) + h2 h2); h / where(nrm > 1e-6, nrm, NaN)
    const nrm = Math.sqrt(hx * hx + hy * hy + hz * hz);
    const d = nrm > hMin ? nrm : NaN;
    hx = hx / d;
    hy = hy / d;
    hz = hz / d;
    // np.cross(u, h): [u1 h2 - u2 h1, u2 h0 - u0 h2, u0 h1 - u1 h0]
    const sx = uy * hz - uz * hy;
    const sy = uz * hx - ux * hz;
    const sz = ux * hy - uy * hx;
    roll[t] = A[iwx + t] * hx + A[iwx + T + t] * hy + A[iwx + 2 * T + t] * hz;
    lat[t] = A[iax + t] * sx + A[iax + T + t] * sy + A[iax + 2 * T + t] * sz;
  }
  const rollC = cumtrapz(roll, dt);
  const latC = cumtrapz(lat, dt);
  const yc = channelIndex('yaw_cum') * T;
  const values: Record<Phys6Name, number> = {
    yaw_q50: at(A, yc, T, 0.5),
    yaw_q100: at(A, yc, T, 1.0),
    roll_q50: at(rollC, 0, T, 0.5),
    roll_q100: at(rollC, 0, T, 1.0),
    latvel_q50: at(latC, 0, T, 0.5),
    latvel_q100: at(latC, 0, T, 1.0),
  };
  if (!PHYS6_FEATURES.every((k) => Number.isFinite(values[k]))) return { values: null, durationS };
  return { values, durationS };
}

export interface Phys6Details {
  readonly output: ModelAttemptOutput;
  readonly features: Phys6Features;
}

export function phys6Forward(m: Phys6, window: ApproachWindow): Phys6Details {
  const features = phys6Features(window, m.hMin);
  const v = features.values;
  if (v === null) return { output: NAN_OUT('phys6'), features };
  // m1_models.Prep._winsor and transform (m1_models.py:185-193): (winsor(x) - c) / s; then decision_function
  // (fit_intercept False: intercept 0)
  let acc = 0.0;
  for (let j = 0; j < m.cols.length; j++) {
    const x = v[PHYS6_FEATURES[m.cols[j]]];
    let w: number;
    if (m.winsor === 'asinh') w = m.med[j] + m.sw[j] * Math.asinh((x - m.med[j]) / m.sw[j]);
    else if (m.winsor === 'clip') w = npClip(x, m.lo[j], m.hi[j]);
    else w = x;
    acc += ((w - m.centre[j]) / m.scale[j]) * m.coef[j];
  }
  const zRaw = acc + m.intercept;
  const p = m.calibrate(zRaw);
  const z = logitRatio(p, m.logitEps);
  return { output: { modelId: 'phys6', zRaw, pRight: p, z, available: true }, features };
}
