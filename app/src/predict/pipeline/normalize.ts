/**
 * norm_v2 as stream_from_record applies it (motionlab/conventions.py device_signals :893-906,
 * normalize_acceleration :740-744, rotation_rate_to_device / RR_MAPPINGS :311-332) and the production
 * normalisation of the page (analysis/stage3/reference.py production_normalization).
 *
 * Production has no calibration: the gravity sign is the engine prior (ENGINE_GRAVITY_SIGN, conventions.py:476:
 * WebKit reports acc and accg negated on every axis; blink, gecko and unknown engines +1), rotationRate is
 * W3C [alpha, beta, gamma] = rates about device [x, y, z] in deg/s, no gyro bias, no iOS quantisation. A wrong
 * sign is a parity transformation that makes a screen-towards-face hold look screen-away, so lane m4's grip
 * gate refuses instead of answering the wrong hand (PORT_SPEC 1.2).
 */
import type { BrowserEngine, RateMappingVerdict, Rows3, StreamNormalization } from '../types';
import { DEG2RAD, RAD2DEG } from './constants';

/** conventions.ENGINE_GRAVITY_SIGN plus the default +1 for an unknown engine (conventions.py:476, 736-737). */
export const ENGINE_GRAVITY_SIGN: Readonly<Record<BrowserEngine, 1 | -1>> = { webkit: -1, blink: 1, gecko: 1, unknown: 1 };

/**
 * The page's normalisation of an engine (reference.production_normalization): sign = the engine prior, W3C
 * deg/s, no quantisation. Like the Python, `engine` is kept only when it is a known engine (null otherwise), which
 * only matters for the repeated-sample rule (WebKit or not).
 */
export function productionNormalization(engine: BrowserEngine | null): StreamNormalization {
  const e: BrowserEngine | null = engine === 'webkit' || engine === 'blink' || engine === 'gecko' ? engine : null;
  return {
    gravitySign: ENGINE_GRAVITY_SIGN[engine ?? 'unknown'],
    rrVerdict: 'w3c',
    rrUnits: 'deg_per_s',
    engine: e,
    quantizeIos: false,
    source: 'engine_prior',
  };
}

/** conventions.RR_MAPPINGS: verdict -> (sign, the rotationRate column [alpha, beta, gamma] giving device x, y, z) */
export const RR_MAPPINGS: Readonly<Record<RateMappingVerdict, readonly [1 | -1, readonly [number, number, number]]>> = {
  w3c: [1, [0, 1, 2]],
  w3c_inverted: [-1, [0, 1, 2]],
  legacy: [1, [1, 2, 0]],
  legacy_inverted: [-1, [1, 2, 0]],
};

/** acc or accg times the session sign (normalize_acceleration: a * float(sign)). */
export function signRows(x: Rows3, sign: 1 | -1): Rows3 {
  const out = new Float64Array(3 * x.n);
  const s = sign === -1 ? -1.0 : 1.0;
  for (let k = 0; k < 3 * x.n; k++) out[k] = x.data[k] * s;
  return { n: x.n, data: out };
}

/** rotation_rate_to_device: rr[..., cols] * (sign * (RAD2DEG if rad/s else 1.0)), deg/s about device x, y, z. */
export function rotationRateToDevice(rr: Rows3, verdict: RateMappingVerdict, units: StreamNormalization['rrUnits']): Rows3 {
  const [sign, cols] = RR_MAPPINGS[verdict] ?? RR_MAPPINGS.w3c;
  const k = sign * (units === 'rad_per_s' ? RAD2DEG : 1.0);
  const out = new Float64Array(3 * rr.n);
  for (let i = 0; i < rr.n; i++) {
    for (let c = 0; c < 3; c++) out[3 * i + c] = rr.data[3 * i + cols[c]] * k;
  }
  return { n: rr.n, data: out };
}

// ───────────────────────────── quantize_like_chrome (stage2 QUANTIZE_IOS; off in production) ─────────────────────────────

const CHROME_ACC_MULTIPLE = 0.1;
const CHROME_GYRO_MULTIPLE_RAD = 0.00174532925199432963;

/** preprocess.chrome_round: nearest multiple, halves away from zero, NaN stays NaN. */
function chromeRound(v: number, multiple: number): number {
  const s = v / multiple;
  return v < 0.0 ? -multiple * Math.floor(-s + 0.5) : multiple * Math.floor(s + 0.5);
}

/** preprocess.quantize_like_chrome(x, 'acc' | 'rr') on raw rows (applied before the sign, like the Python). */
export function quantizeLikeChrome(x: Rows3, kind: 'acc' | 'rr'): Rows3 {
  const out = new Float64Array(3 * x.n);
  for (let k = 0; k < 3 * x.n; k++) {
    const v = x.data[k];
    out[k] = kind === 'acc' ? chromeRound(v, CHROME_ACC_MULTIPLE) : chromeRound(v * DEG2RAD, CHROME_GYRO_MULTIPLE_RAD) * RAD2DEG;
  }
  return { n: x.n, data: out };
}
