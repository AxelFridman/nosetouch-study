/**
 * The final model (lane final, 'm3_cal_deep_platt'): a 4-member 1D CNN ensemble on 12 channels of the (17, 64)
 * approach window, then a Platt calibrator, and sum(logit) / tau_k over k attempts.
 *
 * Port of analysis/stage2/lanes/final_infer.py (the numpy reference of final_model.json; it reproduces the fitted
 * pipeline to 2.2e-16), operation by operation in float64:
 *   extended()       final_infer.py:63-67   the window as float32, plus 4 body-frame channels computed in float64
 *                                            from it and rounded to float32 (Math.fround at exactly that point)
 *   body_channels()  final_infer.py:44-60   h = e_z - (e_z.u) u normalised (0 when |h| <= h_min), s = u x h;
 *                                            roll = w.h, pitch = w.s, lat = a.s, fwd = a.h; every 3-term dot product
 *                                            summed (x0 y0 + x1 y1) + x2 y2 as numpy reduces axis 1
 *   logits_cnn()     final_infer.py:110-131 per member: (x - center) / scale; 3 x (conv1d 'same' with zero padding,
 *                                            no bias; (h - mean) / sqrt(var + eps) * weight + bias; max(h, 0));
 *                                            mean over 8 time bins; flatten channel-major (c * 8 + b); dot fc + bias;
 *                                            the ensemble logit is the mean of the members
 *   calibrate()      final_infer.py:144-164 Platt: sigmoid(clip(a z + b, +-40)), p clipped to [1e-4, 1 - 1e-4]
 *   predict_proba()  final_infer.py:167-174 with rest_up_z: a window whose rest pose has up_z below the threshold is
 *                                            un-flipped (times screen_check.flip_signs) BEFORE the body channels
 *
 * Summation orders that numpy fixes are copied (the 3-term dots, the 8-sample bin means: numpy's pairwise sum,
 * checked); the einsum of the convolution and the BLAS dot of the linear layer have no specified order, so the
 * logit can differ from Python by ~1e-14 (the parity tolerance is 1e-6).
 *
 * NOT validated end to end on detected windows: lane final trained and tested the CNN on oracle windows, and the
 * detector, the stopping thresholds and the session aggregation were fitted for lane m4's classifier (lane prod,
 * which would refit them for this model, has not run). PORT_SPEC 3.4.
 */
import type { AttemptInput, FinalModelJson, ModelAttemptOutput } from '../types';
import { makeCalibrator, type Calibrate } from './calibrators';
import { CHANNELS, N_NORM, logitLog1p, npLinspace, pairwiseSum, roundHalfEven } from './numeric';
import type { ExportedCalibrator } from './jsonTypes';

/** final_infer.extended's body-frame channels, in this order */
export const BODY_CHANNELS = ['roll_rate', 'pitch_rate', 'acc_lat', 'acc_fwd'] as const;

interface ConvLayer {
  readonly cout: number;
  readonly cin: number;
  readonly k: number;
  /** (cout, cin, k) row-major */
  readonly w: Float64Array;
  readonly mean: Float64Array;
  /** np.sqrt(bn_var + bn_eps), computed once exactly as numpy does elementwise */
  readonly sd: Float64Array;
  readonly gamma: Float64Array;
  readonly beta: Float64Array;
}

interface Member {
  readonly center: Float64Array;
  readonly scale: Float64Array;
  readonly layers: readonly ConvLayer[];
  readonly fcW: Float64Array;
  readonly fcB: number;
}

/** The final model, unpacked once from final_cnn.json into typed arrays. */
export interface FinalCnn {
  readonly T: number;
  /** the 12 model channels: their row in the 21-row extended layout (0..16 window, 17..20 body) */
  readonly sources: Int32Array;
  readonly members: readonly Member[];
  readonly width: number;
  readonly nBins: number;
  /** final_infer._bins edges: round(linspace(0, T, n_bins + 1)) */
  readonly binEdges: Int32Array;
  readonly hMin: number;
  readonly calibrate: Calibrate;
  readonly screenThreshold: number;
  /** screen_check.flip_signs as float32 (17) */
  readonly flipSigns: Float32Array;
  readonly tau: Readonly<Record<string, number>>;
}

/** Every intermediate of one attempt (the parity tests compare them with the Python reference). */
export interface CnnDetails {
  readonly output: ModelAttemptOutput;
  readonly memberLogits: readonly number[];
  /** (4, 64) float32, BODY_CHANNELS order */
  readonly body: Float32Array;
  /** the screen check un-flipped the window first */
  readonly unflipped: boolean;
}

const f64 = (a: readonly number[]): Float64Array => Float64Array.from(a);

function fail(msg: string): never {
  throw new RangeError(`final_cnn.json: ${msg}`);
}

/** Unpacks and checks final_cnn.json (the 'nosetouch-final-1' document lane final exported). */
export function buildFinalCnn(doc: FinalModelJson): FinalCnn {
  if (doc.format !== 'nosetouch-final-1') fail(`format ${JSON.stringify(doc.format)}`);
  const m = doc.model;
  if (m.kind !== 'cnn1d_ensemble') fail(`model kind ${JSON.stringify(m.kind)}`);
  const T = doc.input.n_samples;
  if (T !== N_NORM) fail(`n_samples ${T}`);
  const inCh = doc.input.channels;
  if (inCh.length !== CHANNELS.length || inCh.some((c, i) => c !== CHANNELS[i])) {
    fail('input channels differ from standardize.CHANNELS (the app window layout)');
  }
  const body = doc.input.body_frame.channels;
  if (body.length !== 4 || body.some((c, i) => c !== BODY_CHANNELS[i])) fail('body_frame channels');
  const names: string[] = [...inCh, ...body];
  const sources = Int32Array.from(
    m.channels.map((c) => {
      const i = names.indexOf(c);
      if (i < 0) fail(`model channel ${JSON.stringify(c)}`);
      return i;
    }),
  );
  const nCh = sources.length;
  const width = m.width;
  const nBins = m.n_bins;
  // final_infer._bins: e = np.round(np.linspace(0, T, n_bins + 1)).astype(int)
  const binEdges = Int32Array.from(npLinspace(0, T, nBins + 1).map(roundHalfEven));
  const members: Member[] = m.members.map((mem, mi) => {
    if (mem.center.length !== nCh || mem.scale.length !== nCh) fail(`member ${mi}: center/scale length`);
    let cin = nCh;
    const layers: ConvLayer[] = mem.layers.map((L, li) => {
      const cout = L.weight.length;
      const k = L.weight[0]?.[0]?.length ?? 0;
      if (k % 2 !== 1) fail(`member ${mi} layer ${li}: kernel ${k} is not odd`);
      const w = new Float64Array(cout * cin * k);
      L.weight.forEach((byIn, o) => {
        if (byIn.length !== cin) fail(`member ${mi} layer ${li}: ${byIn.length} input channels, expected ${cin}`);
        byIn.forEach((taps, c) => {
          if (taps.length !== k) fail(`member ${mi} layer ${li}: kernel length`);
          for (let j = 0; j < k; j++) w[(o * cin + c) * k + j] = taps[j];
        });
      });
      for (const v of [L.bn_weight, L.bn_bias, L.bn_mean, L.bn_var]) {
        if (v.length !== cout) fail(`member ${mi} layer ${li}: batch-norm length`);
      }
      const sd = new Float64Array(cout);
      for (let o = 0; o < cout; o++) sd[o] = Math.sqrt(L.bn_var[o] + L.bn_eps);
      const layer: ConvLayer = {
        cout,
        cin,
        k,
        w,
        mean: f64(L.bn_mean),
        sd,
        gamma: f64(L.bn_weight),
        beta: f64(L.bn_bias),
      };
      cin = cout;
      return layer;
    });
    if (cin !== width) fail(`member ${mi}: last layer has ${cin} channels, width ${width}`);
    if (mem.fc_weight.length !== width * nBins) fail(`member ${mi}: fc_weight length`);
    return { center: f64(mem.center), scale: f64(mem.scale), layers, fcW: f64(mem.fc_weight), fcB: mem.fc_bias };
  });
  if (members.length === 0) fail('no members');
  const flip = doc.screen_check.flip_signs;
  if (flip.length !== CHANNELS.length) fail('screen_check.flip_signs length');
  return {
    T,
    sources,
    members,
    width,
    nBins,
    binEdges,
    hMin: doc.input.body_frame.h_min,
    calibrate: makeCalibrator(doc.calibrator as unknown as ExportedCalibrator),
    screenThreshold: doc.screen_check.threshold,
    flipSigns: Float32Array.from(flip),
    tau: doc.aggregation.tau,
  };
}

/**
 * final_infer.body_channels on one float32 window, rounded to float32 as final_infer.extended does (:66).
 * Returns (4, T) in BODY_CHANNELS order.
 */
export function finalBodyChannels(A: ArrayLike<number>, T: number, hMin: number): Float32Array {
  const out = new Float32Array(4 * T);
  const iax = 0 * T;
  const iwx = 3 * T;
  const iux = 6 * T;
  for (let t = 0; t < T; t++) {
    const ux = A[iux + t];
    const uy = A[iux + T + t];
    const uz = A[iux + 2 * T + t];
    // h = -u_z u; h_z += 1 (final_infer.py:52-53)
    let hx = -uz * ux;
    let hy = -uz * uy;
    let hz = -uz * uz;
    hz += 1.0;
    const nrm = Math.sqrt(hx * hx + hy * hy + hz * hz);
    // np.where(nrm > h_min, h / nrm, 0.0): a NaN norm also gives 0 (final_infer.py:55-56)
    if (nrm > hMin) {
      hx = hx / nrm;
      hy = hy / nrm;
      hz = hz / nrm;
    } else {
      hx = 0.0;
      hy = 0.0;
      hz = 0.0;
    }
    const sx = uy * hz - uz * hy;
    const sy = uz * hx - ux * hz;
    const sz = ux * hy - uy * hx;
    const wx = A[iwx + t];
    const wy = A[iwx + T + t];
    const wz = A[iwx + 2 * T + t];
    const ax = A[iax + t];
    const ay = A[iax + T + t];
    const az = A[iax + 2 * T + t];
    out[t] = wx * hx + wy * hy + wz * hz;
    out[T + t] = wx * sx + wy * sy + wz * sz;
    out[2 * T + t] = ax * sx + ay * sy + az * sz;
    out[3 * T + t] = ax * hx + ay * hy + az * hz;
  }
  return out;
}

/** One member's logit on the (nCh, T) float64 input (final_infer.logits_cnn's loop body). */
function memberLogit(cnn: FinalCnn, mem: Member, x0: Float64Array): number {
  const T = cnn.T;
  const nCh = cnn.sources.length;
  let h = new Float64Array(nCh * T);
  for (let c = 0; c < nCh; c++) {
    const cc = mem.center[c];
    const sc = mem.scale[c];
    for (let t = 0; t < T; t++) h[c * T + t] = (x0[c * T + t] - cc) / sc;
  }
  for (const L of mem.layers) {
    const { cout, cin, k, w } = L;
    const pad = (k - 1) >> 1;
    const out = new Float64Array(cout * T);
    for (let o = 0; o < cout; o++) {
      const mu = L.mean[o];
      const sd = L.sd[o];
      const g = L.gamma[o];
      const b = L.beta[o];
      for (let t = 0; t < T; t++) {
        // conv1d 'same': out[o, t] = sum_c sum_j W[o, c, j] * h_pad[c, t + j], zero padding (K - 1) / 2
        const j0 = Math.max(0, pad - t);
        const j1 = Math.min(k, T + pad - t);
        let acc = 0.0;
        for (let c = 0; c < cin; c++) {
          const wo = (o * cin + c) * k;
          const ho = c * T + t - pad;
          for (let j = j0; j < j1; j++) acc += w[wo + j] * h[ho + j];
        }
        // batch norm (eval): (h - mean) / sqrt(var + eps) * weight + bias, then ReLU (np.maximum keeps NaN)
        const v = ((acc - mu) / sd) * g + b;
        out[o * T + t] = v > 0 ? v : v === v ? 0.0 : v;
      }
    }
    h = out;
  }
  // _bins: mean over each time bin (np.mean: pairwise sum, then one division); flatten channel-major
  const nb = cnn.nBins;
  const e = cnn.binEdges;
  const prod = new Float64Array(cnn.width * nb);
  for (let c = 0; c < cnn.width; c++) {
    for (let b = 0; b < nb; b++) {
      const n = e[b + 1] - e[b];
      const pooled = pairwiseSum(h, c * T + e[b], n) / n;
      prod[c * nb + b] = pooled * mem.fcW[c * nb + b];
    }
  }
  return pairwiseSum(prod, 0, prod.length) + mem.fcB;
}

/**
 * The CNN on one attempt: final_infer.predict_proba(window[, rest_up_z]) with every intermediate. `available` is
 * false when the logit is not finite (a NaN window), as analysis/stage3/runtime.py ModelSet.cnn_predict decides.
 */
export function cnnForward(cnn: FinalCnn, input: AttemptInput): CnnDetails {
  const T = cnn.T;
  const src = input.window.data;
  if (src.length !== CHANNELS.length * T) throw new RangeError(`window has ${src.length} values, expected ${CHANNELS.length * T}`);
  // final_infer.predict_proba: un-flip only when rest_up_z is given and below the threshold (NaN: no)
  const unflipped = input.restUpZBefore !== undefined && input.restUpZBefore < cnn.screenThreshold;
  let A: Float32Array = src instanceof Float32Array ? src : Float32Array.from(src);
  if (unflipped) {
    const B = new Float32Array(A.length);
    for (let c = 0; c < CHANNELS.length; c++) {
      const s = cnn.flipSigns[c];
      for (let t = 0; t < T; t++) B[c * T + t] = Math.fround(A[c * T + t] * s);
    }
    A = B;
  }
  const body = finalBodyChannels(A, T, cnn.hMin);
  const nCh = cnn.sources.length;
  const x0 = new Float64Array(nCh * T);
  for (let j = 0; j < nCh; j++) {
    const r = cnn.sources[j];
    const from = r < CHANNELS.length ? A : body;
    const off = (r < CHANNELS.length ? r : r - CHANNELS.length) * T;
    for (let t = 0; t < T; t++) x0[j * T + t] = from[off + t];
  }
  const memberLogits = cnn.members.map((mem) => memberLogit(cnn, mem, x0));
  // np.mean(zs, axis=0): the members summed in order, then divided by their number
  const zRaw = pairwiseSum(memberLogits, 0, memberLogits.length) / memberLogits.length;
  if (!Number.isFinite(zRaw)) {
    return {
      output: { modelId: 'cnn', zRaw: NaN, pRight: NaN, z: NaN, available: false },
      memberLogits,
      body,
      unflipped,
    };
  }
  const p = cnn.calibrate(zRaw);
  const z = logitLog1p(p, 1e-12);
  return { output: { modelId: 'cnn', zRaw, pRight: p, z, available: true }, memberLogits, body, unflipped };
}
