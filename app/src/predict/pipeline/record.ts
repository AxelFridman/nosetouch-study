/**
 * TrialRecord.streams.motion -> RawMotion (motionlab/io.py streams_from_record :195-210 with _arr :88-97):
 * every column as float64, null (or a missing / non-numeric value) -> NaN. `ts` and `interval_raw` are not read
 * on the production path (PORT_SPEC 1.1), so they are not carried.
 */
import type { PipelineApi, RawMotion, Rows3 } from '../types';

type MotionColumns = Parameters<PipelineApi['rawMotionFromRecord']>[0];

function num(v: number | null | undefined): number {
  return typeof v === 'number' ? v : NaN;
}

function rows(n: number, x: readonly (number | null)[], y: readonly (number | null)[], z: readonly (number | null)[], what: string): Rows3 {
  if (x.length !== n || y.length !== n || z.length !== n) {
    throw new RangeError(`motion column ${what} has ${x.length}/${y.length}/${z.length} values, t has ${n}`);
  }
  const data = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[3 * i] = num(x[i]);
    data[3 * i + 1] = num(y[i]);
    data[3 * i + 2] = num(z[i]);
  }
  return { n, data };
}

export function rawMotionFromRecord(motion: MotionColumns): RawMotion {
  const n = motion.t.length;
  const t = new Float64Array(n);
  for (let i = 0; i < n; i++) t[i] = num(motion.t[i] as number | null);
  return {
    t,
    acc: rows(n, motion.acc_x, motion.acc_y, motion.acc_z, 'acc'),
    accg: rows(n, motion.accg_x, motion.accg_y, motion.accg_z, 'accg'),
    rr: rows(n, motion.rr_alpha, motion.rr_beta, motion.rr_gamma, 'rr'),
  };
}

/** The first `n` rows of a RawMotion as views (no copy). */
export function rawPrefix(raw: RawMotion, n: number): RawMotion {
  const k = Math.max(0, Math.min(n, raw.t.length));
  if (k === raw.t.length) return raw;
  const view = (r: Rows3): Rows3 => ({ n: k, data: r.data.subarray(0, 3 * k) });
  return { t: raw.t.subarray(0, k), acc: view(raw.acc), accg: view(raw.accg), rr: view(raw.rr) };
}
