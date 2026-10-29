/**
 * Calibrators: the raw logit z of one attempt -> P(right) of that attempt, each model with its own.
 *
 * Python ground truth:
 *   final_infer.calibrate (analysis/stage2/lanes/final_infer.py:144-164): none, temperature, platt, beta, isotonic,
 *     stacked; the logit clip before the sigmoid and the p clip after it. The final CNN ships 'platt'.
 *   m4_models.CALIBRATORS (analysis/stage2/lanes/m4_models.py:250-393): lane m4's 'nested' choice among none,
 *     temperature and isotonic_sym (m4_models.sigmoid clips the logit to +-40). The all-persons fit chose
 *     isotonic_sym.
 *   m1_models._finish (analysis/stage2/lanes/m1_models.py:290-313): PHYS6 'none' = clip(sigmoid(z), 0, 1).
 */
import type { ExportedCalibrator } from './jsonTypes';
import { logitLog1p, npClip, npInterp, sigmoid, sigmoidClipped } from './numeric';

/** z_raw -> calibrated P(right) of one attempt */
export type Calibrate = (zRaw: number) => number;

/** final_infer.logit's eps (final_infer.py:36), used inside the 'stacked' calibrator */
const FINAL_LOGIT_EPS = 1e-12;

/** sigmoid with the optional logit clip of final_infer.sigmoid(z, clip) */
function sig(z: number, logitClip: number | undefined): number {
  return logitClip === undefined || logitClip === null ? sigmoid(z) : sigmoidClipped(z, logitClip);
}

/**
 * m4_models.IsotonicSym.__call__ (m4_models.py:324-328): p = clip(iso.predict(|z|), 0.01, 0.99), P = p for z >= 0,
 * else 1 - p, so P(-z) = 1 - P(z) exactly. iso.predict is scikit-learn's IsotonicRegression._transform: clip T to
 * [X_min_, X_max_], then f_ = scipy interp1d(X_thresholds_, y_thresholds_, 'linear'). scipy 1.17 delegates that
 * interp1d to numpy.interp (interp1d._call_linear_np, checked on the venv), so numpy.interp is the reference here;
 * the searchsorted formula in the export's text gives the same value except at a knot, where it can differ by an ulp.
 */
function isotonicSym(doc: Extract<ExportedCalibrator, { method: 'isotonic_sym' }>): Calibrate {
  const xt = Float64Array.from(doc.x_thresholds);
  const yt = Float64Array.from(doc.y_thresholds);
  if (xt.length === 0 || xt.length !== yt.length) throw new RangeError('isotonic_sym: bad threshold arrays');
  for (let i = 1; i < xt.length; i++) {
    if (!(xt[i] > xt[i - 1])) throw new RangeError('isotonic_sym: x_thresholds must increase strictly');
  }
  const { x_min: xMin, x_max: xMax, p_lo: pLo, p_hi: pHi } = doc;
  return (z: number): number => {
    const T = npClip(Math.abs(z), xMin, xMax);
    // a single threshold is scikit-learn's constant prediction (IsotonicRegression._build_f)
    const iso = xt.length === 1 ? yt[0] : npInterp(T, xt, yt);
    const p = npClip(iso, pLo, pHi);
    return z >= 0 ? p : 1.0 - p;
  };
}

/** The calibrator of an exported document (throws on a method this port does not know). */
export function makeCalibrator(doc: ExportedCalibrator): Calibrate {
  switch (doc.method) {
    case 'nested':
      // m4_models.Nested.__call__ = self.inner(z): the choice was made at fit time
      return makeCalibrator(doc.inner);
    case 'stacked': {
      // final_infer.py:146-148: the inner calibrator, then the outer one on its logit (eps 1e-12); returns early,
      // so no p clip of the stacked entry itself
      const inner = makeCalibrator(doc.inner);
      const outer = makeCalibrator(doc.outer);
      return (z) => outer(logitLog1p(inner(z), FINAL_LOGIT_EPS));
    }
    case 'isotonic_sym':
      return isotonicSym(doc);
    case 'none':
    case 'temperature':
    case 'platt':
    case 'beta':
    case 'isotonic': {
      const raw = rawCalibrator(doc);
      const lo = doc.p_clip;
      // final_infer.py:163-164: np.clip(p, lo, 1 - lo) if lo else p
      const clipped: Calibrate = lo ? (z) => npClip(raw(z), lo, 1 - lo) : raw;
      if (doc.method === 'none' && doc.p_clip_out) {
        const [a, b] = doc.p_clip_out;
        return (z) => npClip(clipped(z), a, b);
      }
      return clipped;
    }
    default: {
      const unknown: never = doc;
      throw new RangeError(`unknown calibrator ${JSON.stringify((unknown as { method?: unknown }).method)}`);
    }
  }
}

function rawCalibrator(doc: Extract<ExportedCalibrator, { method: 'none' | 'temperature' | 'platt' | 'beta' | 'isotonic' }>): Calibrate {
  switch (doc.method) {
    case 'none':
      return (z) => sig(z, doc.logit_clip);
    case 'temperature': {
      const T = doc.T;
      return (z) => sig(z / T, doc.logit_clip);
    }
    case 'platt': {
      const { a, b } = doc;
      return (z) => sig(a * z + b, doc.logit_clip);
    }
    case 'beta': {
      // final_infer.py:156-158
      const { a, b, c } = doc;
      const p0c = doc.p0_clip ?? 0.0;
      return (z) => {
        const p0 = npClip(sigmoid(z), p0c, 1 - p0c);
        return sigmoid(a * Math.log(p0) - b * Math.log1p(-p0) + c);
      };
    }
    case 'isotonic': {
      const x = Float64Array.from(doc.x);
      const y = Float64Array.from(doc.y);
      return (z) => npInterp(z, x, y);
    }
  }
}

/** The method the chosen calibrator applies (the inner method of a nested choice). */
export function effectiveMethod(doc: ExportedCalibrator): string {
  return doc.method === 'nested' ? effectiveMethod(doc.inner) : doc.method;
}
