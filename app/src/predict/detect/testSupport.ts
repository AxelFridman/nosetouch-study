/**
 * TEST SUPPORT ONLY (imported by detect/*.test.ts, never by the page): the golden fixtures, the exported m4 bundle
 * and a minimal m4 hand model, so that detect/ is checked end to end against the Python reference before models/ is
 * wired.
 *
 * Fixtures: the committed synthetic ones (app/src/predict/__fixtures__, format 'nosetouch-fixture-1') and, when the
 * folder exists on this machine, the real ones (analysis/data/stage3_fixtures, git-ignored, absent in CI).
 */
import { existsSync, readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import type { ApproachWindow, DetectorBundle, HandModel, ModelAttemptOutput, ModelId, Rows3, SegmentClass, Stream } from '../types';
import { npSum, searchsortedLeft, sigmoidClipped } from './numeric';

export const SYNTH_DIR = new URL('../__fixtures__/', import.meta.url);
export const REAL_DIR = new URL('../../../../analysis/data/stage3_fixtures/', import.meta.url);
const M4_JSON = new URL('../models/m4_production.json', import.meta.url);

export const realFixturesPresent = (): boolean => existsSync(new URL('index.json', REAL_DIR));

type Num = number | null;

export interface FixtureModelOutput {
  available: boolean;
  z_raw?: Num;
  p?: Num;
  z?: Num;
  p_attempt?: Num;
}

export interface FixtureSegment {
  seg: number;
  onset_ms: number;
  end_ms: number;
  decided_ms: number;
  i_on: number;
  i_end: number;
  still_before_s: number;
  still_after_s: number;
  started_active: boolean;
  ended: boolean;
  peak_dps: number;
  canonical_ok: boolean;
  crop: { start_ms: number; stop_ms: number; n?: number; t0_ms?: number; i_on?: number; i_end?: number };
  gravity?: { method: string; pre_n: number; post_n: number; closure_deg: Num; magnitude: Num };
  features: Record<string, Num>;
  proba: Record<SegmentClass, number>;
  cls: SegmentClass;
  is_attempt: boolean;
  live_frozen: boolean;
  freeze_after_ms: number;
  window?: Num[][];
  models?: Partial<Record<ModelId, FixtureModelOutput>>;
}

export interface FixtureDecision {
  p_right: number;
  percent_right: number;
  stopped: string;
  answered: boolean;
  final: boolean;
  n_used: number;
  decided_ms: Num;
  n_unavailable: number;
  k_curve: { k: number; p: number; seg: number; decided_ms: number }[];
}

export interface Fixture {
  id: string;
  kind: string;
  tags: string[];
  engine: string;
  reference: {
    n_grid: number;
    stream: { t: number[]; f: Num[][]; acc: Num[][]; gyro: Num[][]; valid: boolean[] };
    activity: number[];
    detector: { causal_delay_ms: number; dt_ms: number; n_end: number; n_look: number; n_move: number; seg_cfg: string };
    segments: FixtureSegment[];
    attempts: number[];
    grip: {
      rest_up_z: Num;
      gate_upz: number;
      grip_warning: boolean;
      n_sel: number;
      fallback: boolean;
      first_onset_ms: Num;
      n_default_bouts: number;
    };
    decisions: Record<ModelId, FixtureDecision>;
    m4_predict_stream: {
      p_right: number;
      percent_right: number;
      answered: boolean;
      status: string;
      n_attempts_used: number;
      decided_ms: Num;
      rest_up_z: Num;
      grip_warning: boolean;
      attempts: { onset_ms: number; end_ms: number; decided_ms: number; p_approach: number; p_right: number }[];
      segments: { onset_ms: number; end_ms: number; decided_ms: number; p_approach: number; p_return: number; p_other: number }[];
    };
    t_last_ms: number;
  };
  live: { attempts: number[]; settle_ms: number; decisions: Record<ModelId, FixtureDecision> };
}

export function loadFixtures(dir: URL): Fixture[] {
  const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8')) as { cases: { file: string }[] };
  return index.cases.map(
    (c) => JSON.parse(gunzipSync(readFileSync(new URL(c.file, dir))).toString('utf8')) as Fixture,
  );
}

export function loadM4Json(): { detector: DetectorBundle } & Record<string, unknown> {
  return JSON.parse(readFileSync(M4_JSON, 'utf8')) as { detector: DetectorBundle } & Record<string, unknown>;
}

export function loadDetector(): DetectorBundle {
  return loadM4Json().detector;
}

const num = (v: Num | undefined): number => (v === null || v === undefined ? NaN : v);

function rows3(rows: Num[][]): Rows3 {
  const data = new Float64Array(rows.length * 3);
  rows.forEach((r, i) => {
    for (let j = 0; j < 3; j++) data[3 * i + j] = num(r === null ? null : r[j]);
  });
  return { n: rows.length, data };
}

/** The fixture's standardised stream (reference.stream; null -> NaN) as a Stream. */
export function streamOf(fx: Fixture): Stream {
  const s = fx.reference.stream;
  const t = Float64Array.from(s.t);
  return {
    t,
    f: rows3(s.f),
    acc: rows3(s.acc),
    gyro: rows3(s.gyro),
    valid: Uint8Array.from(s.valid.map((v) => (v ? 1 : 0))),
    fsHz: 60,
    engine: null,
    repeatsDropped: 0,
    tRaw: t,
    gravityMagnitude: null,
    baselineGravity: null,
  };
}

export const numOf = num;

/** A fixture segment's window (17 x 64, null -> NaN) as float32, channel-major. */
export function windowOf(seg: FixtureSegment): Float32Array {
  const w = new Float32Array(17 * 64);
  (seg.window ?? []).forEach((row, c) => row.forEach((v, k) => (w[c * 64 + k] = num(v))));
  return w;
}

/** Distance in float32 ulps between two float32 values (0 when both NaN). */
export function f32Ulps(a: number, b: number): number {
  if (Number.isNaN(a) && Number.isNaN(b)) return 0;
  if (Number.isNaN(a) || Number.isNaN(b)) return Infinity;
  const buf = new DataView(new ArrayBuffer(8));
  const ord = (x: number): number => {
    buf.setFloat32(0, x);
    const i = buf.getInt32(0);
    return i < 0 ? -2147483648 - i : i; // monotone mapping of the float32 bit pattern
  };
  return Math.abs(ord(a) - ord(b));
}

// ───────────────────────────── a minimal m4 hand model (test only) ─────────────────────────────

interface M4HandJson {
  hand: { index: number; step: number; n: number; coef: number[]; intercept: number };
  calibrator: { chosen: string; inner: { x_thresholds: number[]; y_thresholds: number[]; x_min: number; x_max: number; p_lo: number; p_hi: number } };
  aggregation: { s: Record<string, number> };
  logitEps: number;
  sigmoidClip: number;
}

/**
 * Lane m4's hand model from m4_production.json, written from the exported formulas only (yaw8 logistic,
 * isotonic_sym evaluated as scipy interp1d on |z| and reflected, P_k = sigmoid(s_k * mean z)). models/ owns the
 * real one; this copy lets the detect tests check predict_stream's decisions end to end.
 */
export function m4TestModel(): HandModel {
  const j = loadM4Json() as unknown as M4HandJson;
  const { coef, intercept, index, step, n } = j.hand;
  const cal = j.calibrator.inner;
  if (j.calibrator.chosen !== 'isotonic_sym') throw new Error(`m4TestModel: calibrator ${j.calibrator.chosen} not covered`);
  const xs = cal.x_thresholds;
  const ys = cal.y_thresholds;
  const iso = (T: number): number => {
    if (xs.length === 1) return ys[0];
    let k = searchsortedLeft(xs, T);
    k = Math.min(Math.max(k, 1), xs.length - 1);
    return ((ys[k] - ys[k - 1]) / (xs[k] - xs[k - 1])) * (T - xs[k - 1]) + ys[k - 1];
  };
  const calibrate = (z: number): number => {
    const a = Math.abs(z);
    const T = a < cal.x_min ? cal.x_min : a > cal.x_max ? cal.x_max : a;
    let p = iso(T);
    p = p < cal.p_lo ? cal.p_lo : p > cal.p_hi ? cal.p_hi : p;
    return z >= 0 ? p : 1.0 - p;
  };
  const eps = j.logitEps;
  const logit = (p: number): number => {
    const q = p < eps ? eps : p > 1 - eps ? 1 - eps : p;
    return Math.log(q / (1 - q));
  };
  const s = j.aggregation.s;
  const kMaxKey = Math.max(...Object.keys(s).map(Number));
  return {
    spec: { id: 'm4' } as HandModel['spec'],
    predict(input: { window: ApproachWindow }): ModelAttemptOutput {
      let zRaw = 0;
      for (let i = 0; i < n; i++) {
        const v = input.window.data[index * 64 + i * step];
        const x = Number.isNaN(v) ? 0 : v === Infinity ? Number.MAX_VALUE : v === -Infinity ? -Number.MAX_VALUE : v; // nan_to_num
        zRaw += coef[i] * x;
      }
      zRaw += intercept;
      const p = calibrate(zRaw);
      return { modelId: 'm4', zRaw, pRight: p, z: logit(p) };
    },
    aggregate(outputs: readonly ModelAttemptOutput[]): number {
      const k = outputs.length;
      if (k === 0) return 0.5;
      const sk = s[String(k)] ?? s[String(kMaxKey)];
      return sigmoidClipped(sk * (npSum(outputs.map((o) => o.z)) / k), j.sigmoidClip);
    },
  };
}

/** A HandModel whose P_k is read from a fixture decision's k-curve (to test the stopping rule alone). */
export function curveModel(id: ModelId, kCurve: readonly number[]): HandModel {
  return {
    spec: { id } as HandModel['spec'],
    predict(): ModelAttemptOutput {
      throw new Error('curveModel.predict');
    },
    aggregate(outputs: readonly ModelAttemptOutput[]): number {
      return outputs.length === 0 ? 0.5 : kCurve[outputs.length - 1];
    },
  };
}
