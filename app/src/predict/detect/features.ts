/**
 * Hand-agnostic segment features of one bout from its Canonical (m4_detect.segment_features and _kinematics,
 * analysis/stage2/lanes/m4_detect.py:259-324): every name of m4_detect.ALL_FEATURES. The production segment
 * classifier reads the 35 of the 'robust' set (PHYSICS_FEATURES + still_before_inst + still_after; their order comes
 * from the exported bundle, never from this file).
 *
 * Summation follows numpy: np.sum / np.mean of a 1-D slice is numpy's pairwise sum (npSum); reductions over the
 * rows of an (n, 3) array along axis 0 are left-to-right sums.
 */
import type { Bout, Canonical } from '../types';
import { CH, integrateGyro } from './canonical';
import { npArgmax, npMax, npMean, npMin, npSum, pyMax, pyMin, roundHalfEven } from './numeric';

/** m4_detect.INSTRUCTED_STILLNESS_S: the 'robust' set sees the stillness before a bout up to this (s). */
export const INSTRUCTED_STILLNESS_S = 1.0;
const FS_HZ = 60.0; // stage2/config.py:110 (rot_total = sum(|w|) / FS_HZ)

const KIN_NAMES = ['disp_vert', 'disp_y0', 'disp_z0', 'abs_disp_x0', 'disp_norm', 'v_up_peak', 'v_down_peak'] as const;

/** m4_detect.ALL_FEATURES (m4_detect.py:208-229): PHYSICS, CONTEXT, ROBUST_CONTEXT and the GRIP copies, deduplicated. */
export const ALL_FEATURES: readonly string[] = [
  'dur_s', 'log_dur', 'peak_w', 'mean_w', 'rot_total', 'peak_a', 'mean_a', 'abs_net_yaw', 'tilt_end',
  'peak_tilt_rate', 'peak_abs_yaw_rate', 't_peak_w', 'w_first_frac', 't_peak_a', 'a_last_frac', 'a_first_frac',
  'up_y0', 'up_z0', 'abs_up_x0', 'up_y1', 'up_z1', 'abs_up_x1', 'd_up_y', 'd_up_z', 'd_abs_up_x',
  'disp_vert', 'disp_y0', 'disp_z0', 'abs_disp_x0', 'disp_norm', 'v_up_peak', 'v_down_peak', 'closure',
  'still_before', 'still_after', 'started_active', 'seg_index', 'prev_dur', 'still_before_inst',
  'g_up_z0', 'g_up_z1', 'g_d_up_z', 'g_disp_z0',
];

/** m4_detect._kinematics: ZUPT double integration of the linear acceleration over [i_on, i_end]. */
function kinematics(can: Canonical): Record<(typeof KIN_NAMES)[number], number> {
  const { X, n: cn } = can;
  const i0 = can.iOn;
  const i1 = can.iEnd;
  const m = i1 - i0 + 1;
  const nan = Object.fromEntries(KIN_NAMES.map((k) => [k, NaN])) as Record<(typeof KIN_NAMES)[number], number>;
  const ts = new Float64Array(m); // t = can.t[i0:i1+1] / 1000.0 (s)
  for (let k = 0; k < m; k++) ts[k] = can.t[i0 + k] / 1000.0;
  const a = new Float64Array(3 * m);
  const w = new Float64Array(3 * m);
  let finite = true;
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < 3; j++) {
      const av = X[(CH.acc_x + j) * cn + i0 + k];
      const wv = X[(CH.gyro_x + j) * cn + i0 + k];
      a[3 * k + j] = av;
      w[3 * k + j] = wv;
      if (!Number.isFinite(av) || !Number.isFinite(wv)) finite = false;
    }
  }
  if (m < 3 || !finite) return nan; // m4_detect.py:268-270
  const u0 = [X[CH.up_x * cn + i0], X[CH.up_y * cn + i0], X[CH.up_z * cn + i0]];
  // integrate_gyro(t * 1000.0, w): exactly this expression ((t_ms / 1000) * 1000, divided again inside)
  const tms = new Float64Array(m);
  for (let k = 0; k < m; k++) tms[k] = ts[k] * 1000.0;
  const R = integrateGyro(tms, w, 0, m); // device(k) -> device(onset)
  const aw = new Float64Array(3 * m);
  for (let k = 0; k < m; k++) {
    const o = 9 * k;
    const x0 = a[3 * k];
    const x1 = a[3 * k + 1];
    const x2 = a[3 * k + 2];
    aw[3 * k] = R[o] * x0 + R[o + 1] * x1 + R[o + 2] * x2;
    aw[3 * k + 1] = R[o + 3] * x0 + R[o + 4] * x1 + R[o + 5] * x2;
    aw[3 * k + 2] = R[o + 6] * x0 + R[o + 7] * x1 + R[o + 8] * x2;
  }
  // v = cumulative_trapezoid(aw, t, axis=0, initial=0): v[k+1] = v[k] + (dt * (aw[k+1] + aw[k])) / 2.0
  const v = new Float64Array(3 * m);
  for (let k = 0; k < m - 1; k++) {
    const d = ts[k + 1] - ts[k];
    for (let j = 0; j < 3; j++) v[3 * (k + 1) + j] = v[3 * k + j] + (d * (aw[3 * (k + 1) + j] + aw[3 * k + j])) / 2.0;
  }
  const T = ts[m - 1] - ts[0];
  if (T > 0) {
    // ZUPT: v -= outer((t - t0) / T, v[-1]) (zero velocity at both rests)
    const vl = [v[3 * (m - 1)], v[3 * (m - 1) + 1], v[3 * (m - 1) + 2]];
    for (let k = 0; k < m; k++) {
      const r = (ts[k] - ts[0]) / T;
      for (let j = 0; j < 3; j++) v[3 * k + j] = v[3 * k + j] - r * vl[j];
    }
  }
  // dp = trapezoid(v, t, axis=0): a left-to-right sum over k of (dt * (v[k+1] + v[k])) / 2.0
  const dp = [0, 0, 0];
  for (let k = 0; k < m - 1; k++) {
    const d = ts[k + 1] - ts[k];
    for (let j = 0; j < 3; j++) dp[j] += (d * (v[3 * (k + 1) + j] + v[3 * k + j])) / 2.0;
  }
  const vv = new Float64Array(m);
  for (let k = 0; k < m; k++) vv[k] = v[3 * k] * u0[0] + v[3 * k + 1] * u0[1] + v[3 * k + 2] * u0[2];
  return {
    disp_vert: dp[0] * u0[0] + dp[1] * u0[1] + dp[2] * u0[2],
    disp_y0: dp[1],
    disp_z0: dp[2],
    abs_disp_x0: Math.abs(dp[0]),
    disp_norm: Math.sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]),
    v_up_peak: pyMax(0.0, npMax(vv)), // Python max(0.0, nan) is 0.0
    v_down_peak: pyMax(0.0, -npMin(vv)),
  };
}

/** m4_detect.SCREEN_ODD_FEATURES: their 'grip' copies g_<name> are multiplied by sign(up_z0 + up_z1). */
const SCREEN_ODD_FEATURES = ['up_z0', 'up_z1', 'd_up_z', 'disp_z0'] as const;

/**
 * m4_detect.segment_features(can, seg, prev): every name of ALL_FEATURES -> value (NaN where not computable).
 * `bout` supplies the bounds and the stillness before / after; `prev` is the previous bout of the stream (only
 * prev_dur reads it; the production 'robust' set reads neither prev_dur, seg_index nor the g_* copies).
 */
export function segmentFeatures(can: Canonical, bout: Bout, prev: Bout | null = null): Record<string, number> {
  const { X, n: cn } = can;
  const i0 = can.iOn;
  const i1 = can.iEnd;
  const m = i1 - i0 + 1;
  const ch = (c: keyof typeof CH): Float64Array => X.subarray(CH[c] * cn + i0, CH[c] * cn + i1 + 1);
  const t = can.t.subarray(i0, i1 + 1);
  const dur = pyMax((bout.endMs - bout.onsetMs) / 1000.0, 1e-3);
  const wm = ch('gyro_mag');
  const am = ch('acc_mag');
  const span = pyMax(t[m - 1] - t[0], 1e-9);
  const tt = new Float64Array(m);
  for (let k = 0; k < m; k++) tt[k] = (t[k] - t[0]) / span;
  const sumW = npSum(wm);
  const totW = sumW === 0 ? 1e-9 : sumW; // `float(np.sum(wm)) or 1e-9`: only an exact 0 is replaced (NaN is truthy)
  const halfW: number[] = [];
  for (let k = 0; k < m; k++) if (tt[k] <= 0.5) halfW.push(wm[k]);
  const q = Math.max(1, roundHalfEven(0.2 * m));
  const meanA = npMean(am);
  const amean = meanA === 0 ? 1e-9 : meanA;
  const up = (k: number): [number, number, number] => [X[CH.up_x * cn + k], X[CH.up_y * cn + k], X[CH.up_z * cn + k]];
  const up0 = up(i0);
  const up1 = up(i1);
  const yaw = ch('yaw_cum');
  const yawRate = ch('yaw_rate');
  const absYawRate = new Float64Array(m);
  for (let k = 0; k < m; k++) absYawRate[k] = Math.abs(yawRate[k]);
  const f: Record<string, number> = {
    dur_s: dur,
    log_dur: Math.log(dur),
    peak_w: npMax(wm),
    mean_w: npMean(wm),
    rot_total: npSum(wm) / FS_HZ,
    peak_a: npMax(am),
    mean_a: amean,
    abs_net_yaw: Math.abs(yaw[m - 1] - yaw[0]),
    tilt_end: ch('tilt_from_start')[m - 1],
    peak_tilt_rate: npMax(ch('gyro_horiz')),
    peak_abs_yaw_rate: npMax(absYawRate),
    t_peak_w: tt[npArgmax(wm)],
    w_first_frac: npSum(halfW) / totW,
    t_peak_a: tt[npArgmax(am)],
    a_last_frac: npMean(am, m - q, q) / amean,
    a_first_frac: npMean(am, 0, q) / amean,
    up_y0: up0[1],
    up_z0: up0[2],
    abs_up_x0: Math.abs(up0[0]),
    up_y1: up1[1],
    up_z1: up1[2],
    abs_up_x1: Math.abs(up1[0]),
    d_up_y: up1[1] - up0[1],
    d_up_z: up1[2] - up0[2],
    d_abs_up_x: Math.abs(up1[0]) - Math.abs(up0[0]),
    closure: Number.isFinite(can.closureDeg) ? can.closureDeg : 0.0,
    still_before: bout.stillBeforeS,
    still_after: bout.stillAfterS,
    still_before_inst: pyMin(bout.stillBeforeS, INSTRUCTED_STILLNESS_S),
    started_active: bout.startedActive ? 1.0 : 0.0,
    seg_index: pyMin(bout.index, 5),
    prev_dur: prev === null ? 0.0 : pyMin((prev.endMs - prev.onsetMs) / 1000.0, 10.0),
    ...kinematics(can),
  };
  // grip-normalised copies of the screen-odd features (m4_detect.py:318-323)
  const tot = f.up_z0 + f.up_z1;
  const s = Number.isFinite(tot) && tot < 0 ? -1.0 : 1.0;
  for (const k of SCREEN_ODD_FEATURES) f[`g_${k}`] = Number.isFinite(f[k]) ? f[k] * s : NaN;
  return f;
}
