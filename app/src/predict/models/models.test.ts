/**
 * Unit tests of the models owner's code: numpy-exact primitives, calibrators, the k-attempt aggregations, model
 * properties (the CNN's screen un-flip, PHYS6's antisymmetry and its unavailable cases, nan_to_num in m4 only),
 * the registry (ids, specs, sha256 of the bundled files, the detector bundle) and the CNN latency per attempt.
 * Parity with Python on the golden fixtures is in parity.test.ts.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import type { ApproachWindow, ModelAttemptOutput, ModelId } from '../types';
import { aggregateOutputs, kCurve, tableAt } from './aggregate';
import { makeCalibrator } from './calibrators';
import { cnnForward, finalBodyChannels } from './finalCnn';
import type { ExportedCalibrator } from './jsonTypes';
import { m4Forward, phys6Features, phys6Forward, yawForward } from './logistic';
import {
  CHANNELS,
  logitLog1p,
  logitRatio,
  nanToNum,
  npInterp,
  npLinspace,
  pairwiseSum,
  roundHalfEven,
  sigmoid,
  sigmoidClipped,
} from './numeric';
import {
  MODEL_FILES,
  MODEL_ORDER,
  THRESHOLDS_NOTE,
  bundle,
  defaultModel,
  detector,
  loadModels,
  modelById,
  modelInfo,
  modelsApi,
  specById,
  specs,
  stoppingParams,
} from './registry';
import cnnJson from './final_cnn.json';
import m4Json from './m4_production.json';
import manifestJson from './manifest.json';

// a realistic window: the first windowed segment of the first committed synthetic fixture
function fixtureWindow(): ApproachWindow {
  const url = new URL('../__fixtures__/syn01_android_right_3touch.json.gz', import.meta.url);
  const fx = JSON.parse(gunzipSync(readFileSync(url)).toString('utf8'));
  const s = fx.reference.segments.find((x: { window?: unknown }) => x.window);
  const data = new Float32Array(17 * 64);
  (s.window as (number | null)[][]).forEach((r, c) => r.forEach((v, t) => (data[c * 64 + t] = v === null ? NaN : v)));
  return { data, onsetMs: s.onset_ms, endMs: s.end_ms };
}

const W0 = fixtureWindow();
const row = (name: string) => CHANNELS.indexOf(name as (typeof CHANNELS)[number]);
const withData = (w: ApproachWindow, f: (d: Float32Array) => void): ApproachWindow => {
  const d = Float32Array.from(w.data);
  f(d);
  return { ...w, data: d };
};
const out = (p: number, z = logitRatio(p, 1e-6), available = true): ModelAttemptOutput => ({
  modelId: 'm4',
  zRaw: z,
  pRight: p,
  z,
  available,
});

describe('numeric primitives (numpy semantics)', () => {
  it('npInterp: edges, knots, interior, NaN, one point', () => {
    const xp = [0, 1, 2];
    const fp = [0, 10, 30];
    expect(npInterp(-1, xp, fp)).toBe(0);
    expect(npInterp(0, xp, fp)).toBe(0);
    expect(npInterp(0.5, xp, fp)).toBe(5);
    expect(npInterp(1, xp, fp)).toBe(10);
    expect(npInterp(1.5, xp, fp)).toBe(20);
    expect(npInterp(2, xp, fp)).toBe(30);
    expect(npInterp(3, xp, fp)).toBe(30);
    expect(npInterp(NaN, xp, fp)).toBeNaN();
    expect(npInterp(5, [1], [7])).toBe(7);
  });

  it('pairwiseSum follows numpy (values checked with numpy 2.4)', () => {
    expect(pairwiseSum([1e16, 1, 1], 0, 3)).toBe(1e16); // np.sum: sequential below 8 terms
    const eight = [1e16, 1, 1, 1, 1, 1, 1, 1];
    expect(pairwiseSum(eight, 0, 8)).toBe(1.0000000000000006e16); // 8 accumulators
    const long = new Float64Array(256);
    long.set(eight);
    expect(pairwiseSum(long, 0, 256)).toBe(1.0000000000000006e16);
    expect(pairwiseSum([2, 4, 6], 0, 3, 1)).toBe(12);
    expect(pairwiseSum([1, 9, 2, 9, 3, 9], 0, 3, 2)).toBe(6);
  });

  it("roundHalfEven is Python's round", () => {
    expect([0.5, 1.5, 2.5, 12.5, -0.5, 87.4, 87.6, 99.5].map(roundHalfEven)).toEqual([0, 2, 2, 12, 0, 87, 88, 100]);
  });

  it('linspace, nan_to_num, logits, sigmoids', () => {
    expect(npLinspace(0, 64, 9)).toEqual([0, 8, 16, 24, 32, 40, 48, 56, 64]);
    expect([NaN, Infinity, -Infinity, 1.5].map(nanToNum)).toEqual([0, Number.MAX_VALUE, -Number.MAX_VALUE, 1.5]);
    expect(Math.abs(logitLog1p(0.9, 1e-12) - Math.log(9))).toBeLessThan(1e-14);
    expect(Math.abs(logitRatio(0.9, 1e-6) - Math.log(9))).toBeLessThan(1e-14);
    expect(logitRatio(1, 1e-6)).toBe(Math.log((1 - 1e-6) / (1 - (1 - 1e-6))));
    expect(logitLog1p(0, 1e-12)).toBe(Math.log(1e-12) - Math.log1p(-1e-12));
    expect(sigmoidClipped(100, 40)).toBe(sigmoid(40));
    expect(sigmoidClipped(-3, 40)).toBe(sigmoid(-3));
  });
});

describe('calibrators', () => {
  it('Platt (the final CNN): sigmoid(clip(a z + b, +-40)), p clipped to [1e-4, 1 - 1e-4]', () => {
    const c = cnnJson.calibrator;
    const cal = makeCalibrator(c as ExportedCalibrator);
    for (const z of [-50, -7, -0.3, 0, 0.2, 3, 9.5, 60]) {
      const want = Math.min(Math.max(sigmoid(Math.min(Math.max(c.a * z + c.b, -40), 40)), 1e-4), 1 - 1e-4);
      expect(cal(z)).toBe(want);
    }
    expect(cal(1000)).toBe(1 - 1e-4);
  });

  it("lane m4's nested choice is isotonic_sym: antisymmetric, exact at the knots, clipped to [0.01, 0.99]", () => {
    const doc = m4Json.calibrator as ExportedCalibrator;
    expect(doc.method).toBe('nested');
    const inner = (m4Json.calibrator as { inner: { x_thresholds: number[]; y_thresholds: number[]; x_max: number } }).inner;
    const cal = makeCalibrator(doc);
    let prev = -Infinity;
    for (let i = -400; i <= 400; i++) {
      const z = i / 20;
      const p = cal(z);
      expect(p).toBeGreaterThanOrEqual(0.01);
      expect(p).toBeLessThanOrEqual(0.99);
      expect(p).toBeGreaterThanOrEqual(prev);
      prev = p;
      if (z > 0) expect(cal(-z)).toBe(1 - p);
    }
    inner.x_thresholds.forEach((x, j) => {
      if (x >= 0) expect(cal(x)).toBe(Math.min(Math.max(inner.y_thresholds[j], 0.01), 0.99));
    });
    expect(cal(inner.x_max * 3)).toBe(0.99);
    expect(cal(-inner.x_max * 3)).toBeCloseTo(0.01, 15);
  });

  it('none / temperature with the m4 logit clip, PHYS6 output clip, beta, plain isotonic and stacked', () => {
    expect(makeCalibrator({ method: 'none', logit_clip: 40 })(100)).toBe(sigmoid(40));
    expect(makeCalibrator({ method: 'none' })(100)).toBe(sigmoid(100));
    expect(makeCalibrator({ method: 'temperature', T: 2, logit_clip: 40 })(3)).toBe(sigmoid(1.5));
    expect(makeCalibrator({ method: 'none', p_clip_out: [0, 1] })(2)).toBe(sigmoid(2));
    const beta = makeCalibrator({ method: 'beta', a: 1, b: 1, c: 0 });
    expect(Math.abs(beta(1.3) - sigmoid(1.3))).toBeLessThan(1e-15);
    expect(makeCalibrator({ method: 'isotonic', x: [-1, 1], y: [0.2, 0.8] })(0)).toBe(0.5);
    const stacked = makeCalibrator({
      method: 'stacked',
      inner: { method: 'none' },
      outer: { method: 'temperature', T: 2 },
    });
    expect(Math.abs(stacked(2) - sigmoid(1))).toBeLessThan(1e-15);
    expect(() => makeCalibrator({ method: 'unknown' } as unknown as ExportedCalibrator)).toThrow();
  });
});

describe('k-attempt aggregations', () => {
  const ps = [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.95];

  it('no usable attempt gives 0.5, for every kind', () => {
    for (const s of specs) {
      expect(aggregateOutputs(s.aggregation, [])).toBe(0.5);
      expect(aggregateOutputs(s.aggregation, [out(0.9, NaN, false)])).toBe(0.5);
    }
  });

  it('sum_over_tau (CNN): sigmoid(sum logit(p, eps 1e-12) / tau_k), tau_1 = 1, beyond the table the largest', () => {
    const tau = { '1': 1.0, '2': 1.25, '3': 1.5 };
    const o = ps.map((p) => out(p, logitLog1p(p, 1e-12)));
    const z = ps.map((p) => logitLog1p(p, 1e-12));
    expect(aggregateOutputs({ kind: 'sum_over_tau', tau }, o.slice(0, 1))).toBe(sigmoid(z[0]));
    expect(aggregateOutputs({ kind: 'sum_over_tau', tau }, o.slice(0, 3))).toBe(sigmoid((z[0] + z[1] + z[2]) / 1.5));
    const s7 = pairwiseSum(z, 0, 7);
    expect(aggregateOutputs({ kind: 'sum_over_tau', tau }, o)).toBe(sigmoid(s7 / 1.5));
  });

  it('scaled_mean (m4): clipped sigmoid of s_k times the mean; s null is the plain mean', () => {
    const s = { '1': 0.8, '2': 1.2, '3': 1.5 };
    const o = ps.map((p) => out(p));
    const z = o.map((x) => x.z);
    expect(aggregateOutputs({ kind: 'scaled_mean', s }, o.slice(0, 2))).toBe(sigmoidClipped(1.2 * ((z[0] + z[1]) / 2), 40));
    expect(aggregateOutputs({ kind: 'scaled_mean', s }, o.slice(0, 5))).toBe(sigmoidClipped(1.5 * (pairwiseSum(z, 0, 5) / 5), 40));
    expect(aggregateOutputs({ kind: 'scaled_mean', s: null }, o.slice(0, 2))).toBe(sigmoidClipped((z[0] + z[1]) / 2, 40));
    expect(aggregateOutputs({ kind: 'scaled_mean', s }, [out(1 - 1e-7, 300)])).toBe(sigmoid(40));
    expect(tableAt(s, 9)).toBe(1.5);
    expect(tableAt({}, 2)).toBeUndefined();
  });

  it('sum_over_k_pow (PHYS6) and mean_p (yaw reference)', () => {
    const o = ps.map((p) => out(p));
    const z = o.map((x) => x.z);
    expect(aggregateOutputs({ kind: 'sum_over_k_pow', alpha: 0.25 }, o.slice(0, 3))).toBe(
      sigmoid((z[0] + z[1] + z[2]) / Math.pow(3, 0.25)),
    );
    expect(aggregateOutputs({ kind: 'mean_p' }, o.slice(0, 3))).toBe((0.8 + 0.7 + 0.9) / 3);
    expect(aggregateOutputs({ kind: 'mean' }, o.slice(0, 2))).toBe(sigmoidClipped((z[0] + z[1]) / 2, 40));
    expect(aggregateOutputs({ kind: 'sum' }, o.slice(0, 2))).toBe(sigmoidClipped(z[0] + z[1], 40));
  });

  it('skips unavailable attempts (that model only) and kCurve lists P_1..P_K', () => {
    const spec = { kind: 'mean_p' } as const;
    const o = [out(0.8), out(NaN, NaN, false), out(0.6)];
    expect(aggregateOutputs(spec, o)).toBe(0.7);
    expect(kCurve(spec, o)).toEqual([0.8, 0.7]);
  });
});

describe('the models on one window', () => {
  const b = bundle();

  it('CNN: the screen check un-flips a screen-away window first (final_infer.predict_proba with rest_up_z)', () => {
    const flipped = withData(W0, (d) => {
      for (let c = 0; c < 17; c++) for (let t = 0; t < 64; t++) d[c * 64 + t] = Math.fround(d[c * 64 + t] * b.cnn.flipSigns[c]);
    });
    const ref = cnnForward(b.cnn, { window: W0 });
    const un = cnnForward(b.cnn, { window: flipped, restUpZBefore: -0.4 });
    expect(un.unflipped).toBe(true);
    expect(un.output).toEqual(ref.output);
    expect(cnnForward(b.cnn, { window: W0, restUpZBefore: 0.4 }).unflipped).toBe(false);
    expect(cnnForward(b.cnn, { window: W0, restUpZBefore: NaN }).unflipped).toBe(false);
    expect(ref.output.available).toBe(true);
    expect(ref.memberLogits).toHaveLength(4);
  });

  it('CNN: a NaN window is unavailable; a horizontal screen gives zero body channels', () => {
    const bad = withData(W0, (d) => d.fill(NaN, 9 * 64, 10 * 64)); // gyro_mag
    const o = cnnForward(b.cnn, { window: bad }).output;
    expect(o.available).toBe(false);
    expect(o.pRight).toBeNaN();
    const flat = withData(W0, (d) => {
      d.fill(0, 6 * 64, 8 * 64);
      d.fill(1, 8 * 64, 9 * 64);
    });
    const body = finalBodyChannels(flat.data, 64, b.cnn.hMin);
    expect(Array.from(body).every((v) => v === 0)).toBe(true);
  });

  it('PHYS6 is exactly antisymmetric: the mirrored window negates z_raw', () => {
    const info = (cnnJson.input as unknown as { channel_info: Record<string, [string, string, number, number]> }).channel_info;
    const mirror = withData(W0, (d) => {
      CHANNELS.forEach((name, c) => {
        const s = info[name][2];
        for (let t = 0; t < 64; t++) d[c * 64 + t] = s * d[c * 64 + t];
      });
    });
    const a = phys6Forward(b.phys6, W0).output;
    const m = phys6Forward(b.phys6, mirror).output;
    expect(a.available).toBe(true);
    expect(m.zRaw).toBe(-a.zRaw);
    // lane m4's yaw8 and isotonic_sym are antisymmetric up to the ~1e-18 intercept
    const pa = m4Forward(b.m4, W0).output.pRight;
    const pm = m4Forward(b.m4, mirror).output.pRight;
    expect(Math.abs(pa + pm - 1)).toBeLessThan(1e-12);
  });

  it('PHYS6 is unavailable with the screen exactly horizontal, a NaN window or no duration', () => {
    const h = withData(W0, (d) => {
      d[6 * 64 + 20] = 0;
      d[7 * 64 + 20] = 0;
      d[8 * 64 + 20] = 1;
    });
    expect(phys6Features(h, b.phys6.hMin).values).toBeNull();
    expect(phys6Forward(b.phys6, h).output.available).toBe(false);
    expect(phys6Forward(b.phys6, withData(W0, (d) => (d[13 * 64 + 5] = NaN))).output.available).toBe(false);
    expect(phys6Forward(b.phys6, { ...W0, endMs: W0.onsetMs }).output.available).toBe(false);
    const ok = phys6Features(W0, b.phys6.hMin);
    expect(ok.durationS).toBe((W0.endMs - W0.onsetMs) / 1000);
    expect(ok.values!.yaw_q100).toBe(W0.data[row('yaw_cum') * 64 + 63]);
  });

  it('yaw reference: unavailable on a NaN sample; m4 maps it to 0 (nan_to_num)', () => {
    const w = withData(W0, (d) => (d[row('yaw_cum') * 64 + 8] = NaN));
    expect(yawForward(b.yaw, w).output.available).toBe(false);
    const m = m4Forward(b.m4, w);
    expect(m.output.available).toBe(true);
    expect(m.x8[1]).toBe(0);
    const z = withData(W0, (d) => (d[row('yaw_cum') * 64 + 8] = 0));
    expect(m.output).toEqual(m4Forward(b.m4, z).output);
    expect(m.pAttempt).toBe(1 / (1 + Math.exp(-m.output.z)));
  });

  it('rejects a window of the wrong size', () => {
    const w = { ...W0, data: new Float32Array(10) };
    expect(() => cnnForward(b.cnn, { window: w })).toThrow();
    expect(() => yawForward(b.yaw, w)).toThrow();
  });
});

describe('registry', () => {
  it('lists cnn, m4, phys6, yaw with the CNN as the default', () => {
    expect(specs.map((s) => s.id)).toEqual(['cnn', 'm4', 'phys6', 'yaw']);
    expect(MODEL_ORDER).toEqual(['cnn', 'm4', 'phys6', 'yaw']);
    expect(defaultModel).toBe('cnn');
    expect(specs.filter((s) => s.isDefault).map((s) => s.id)).toEqual(['cnn']);
    expect(specs.map((s) => s.label)).toEqual(['Final CNN', 'Detector model', 'Physics logistic', 'Yaw reference']);
    expect(specs.map((s) => s.stage2Name)).toEqual(['m3_cal_deep_platt', 'm4_cal_nested', 'm1_lr_phys6_nested', 'example_yaw_logreg']);
    expect(specs.map((s) => s.kind)).toEqual(['cnn1d_ensemble', 'logistic', 'logistic', 'logistic']);
    expect(specs.map((s) => s.calibration)).toEqual(['platt', 'isotonic_sym', 'none', 'none']);
    expect(specs.map((s) => s.aggregation.kind)).toEqual(['sum_over_tau', 'scaled_mean', 'sum_over_k_pow', 'mean_p']);
    for (const s of specs) expect(s.trainedOn).toEqual({ persons: 20, trials: 593 });
    const manifest = manifestJson as unknown as { order: string[]; default: string; detector: string };
    expect(manifest.order).toEqual([...MODEL_ORDER]);
    expect(manifest.default).toBe(defaultModel);
    expect(manifest.detector).toBe('m4');
    expect(specById('phys6').aggregation).toEqual({ kind: 'sum_over_k_pow', alpha: 0.25 });
    expect(() => specById('nope' as ModelId)).toThrow();
  });

  it('carries the sha256 of every bundled file (recomputed from the files on disk)', () => {
    const manifest = manifestJson as unknown as { models: { id: ModelId; sha256: string; bytes: number }[] };
    for (const s of specs) {
      // the committed bytes are LF; a Windows checkout with core.autocrlf may add CR, which is not a model change
      const bytes = readFileSync(new URL(MODEL_FILES[s.id], import.meta.url));
      const lf = Buffer.from(bytes.toString('latin1').replace(/\r\n/g, '\n'), 'latin1');
      expect(createHash('sha256').update(lf).digest('hex'), s.id).toBe(s.sha256);
      expect(manifest.models.find((m) => m.id === s.id)!.sha256).toBe(s.sha256);
      expect(modelInfo[s.id].shortHash).toBe(s.sha256.slice(0, 8));
    }
  });

  it('modelInfo: one line and the Stage 2 LOSO accuracy per model', () => {
    expect(modelInfo.cnn.short).toBe('1D CNN, 4 members');
    for (const id of MODEL_ORDER) {
      const a = modelInfo[id].accuracy!;
      expect(a).toBeGreaterThan(0.5);
      expect(modelInfo[id].ci95![0]).toBeLessThan(a);
      expect(modelInfo[id].ci95![1]).toBeGreaterThan(a);
    }
  });

  it("exposes lane m4's detector and stopping thresholds, noting they were fitted for m4's classifier", () => {
    const d = detector();
    expect(d.segCfg).toBe('lo15_es250');
    expect(d.segParams.lo_dps).toBe(15);
    expect(d.segmentClassifier.classes).toEqual(['approach', 'other', 'return']);
    expect(d.segmentClassifier.features).toHaveLength(35);
    expect(d.lowpassSos).toHaveLength(2);
    expect(d.lowpassPadlen).toBe(15);
    expect(d.grip.gateUpz).toBe(-0.05); // the page's gate (analysis/stage3/runtime.py PRODUCTION_GRIP_GATE_UPZ; lane m4 uses 0)
    expect(stoppingParams()).toEqual({ pStop: 0.9, kMax: 5, pAnswer: 0.5 });
    expect(THRESHOLDS_NOTE).toMatch(/lane prod/);
    expect(modelsApi.detector()).toBe(d);
  });

  it('loadModels gives four pure HandModels in registry order', () => {
    const models = loadModels();
    expect(models.map((m) => m.spec.id)).toEqual(['cnn', 'm4', 'phys6', 'yaw']);
    expect(modelsApi.loadModels().map((m) => m.spec)).toEqual(models.map((m) => m.spec));
    for (const m of models) {
      const o = m.predict({ window: W0 });
      expect(o.modelId).toBe(m.spec.id);
      expect(o.available).toBe(true);
      expect(o.pRight).toBeGreaterThan(0);
      expect(o.pRight).toBeLessThan(1);
      expect(m.predict({ window: W0 })).toEqual(o);
      expect(m.aggregate([o])).toBeGreaterThan(0);
    }
    expect(modelById('yaw').spec.id).toBe('yaw');
    expect(() => modelById('nope' as ModelId)).toThrow();
  });
});

describe('CNN latency', () => {
  it('measures one attempt (4 members) in Node', () => {
    const cnn = bundle().cnn;
    for (let i = 0; i < 20; i++) cnnForward(cnn, { window: W0 }); // warm up the JIT
    const ms: number[] = [];
    for (let i = 0; i < 100; i++) {
      const t0 = performance.now();
      cnnForward(cnn, { window: W0 });
      ms.push(performance.now() - t0);
    }
    ms.sort((a, b) => a - b);
    const med = ms[50];
    const p95 = ms[95];
    console.log(`CNN latency per attempt (Node ${process.version}): median ${med.toFixed(2)} ms, p95 ${p95.toFixed(2)} ms`);
    expect(med).toBeLessThan(250);
  });
});
