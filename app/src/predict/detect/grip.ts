/**
 * The grip check (model-free): m4_detect.rest_up_z (analysis/stage2/lanes/m4_detect.py:424-444) and the gate of
 * m4_production.predict_stream (m4_production.py:104-109; m4_config.GRIP_GATE_UPZ 0.0, GRIP_REST_MIN_S 0.25).
 *
 * rest_up_z = median of f_z / |f| (the screen-normal component of the RAW specific-force direction) over the still
 * samples (activity < lo_dps of the DEFAULT detector, lo 20) before the onset of the stream's first DEFAULT bout
 * (ended or not, lag-corrected); with fewer than round(0.25 s * 60) = 15 such samples, over the first 0.25 s of the
 * stream. Below the gate (screen facing away or down) the stream is not answered (status 'grip', P = 0.5).
 *
 * It always runs with m4_detect.DEFAULT (bundle grip.restParams), which differs from the fitted detector
 * (segParams, lo 15). Both share smooth_hz = 5, hence the bundle's single activity filter.
 */
import type { Bout, DetectorBundle, Stream } from '../types';
import { activity } from './activity';
import { NOMINAL_STEP_MS, boutConstants, findBouts, isFrozen, medianStep } from './bouts';
import { norm3, npMedian, roundHalfEven } from './numeric';

/** The details the debug view and the parity tests read. */
export interface RestDetails {
  /** NaN when the stream has no finite specific force (or no sample) */
  readonly restUpZ: number;
  readonly nSel: number;
  /** the first 0.25 s were used (too few still samples before the first movement) */
  readonly fallback: boolean;
  /** onset of the first DEFAULT bout (ms), +Infinity when there is none */
  readonly firstOnsetMs: number;
  readonly nDefaultBouts: number;
  /** the first DEFAULT bout (null when there is none) */
  readonly firstBout: Bout | null;
  /** its look-ahead in samples (the freeze rule of a live prefix) */
  readonly nLook: number;
}

/** The DEFAULT detector's causal delay: exported, or the fitted detector's when both filters are the same. */
function restLagMs(detector: DetectorBundle): number {
  const exported = detector.grip.restCausalDelayMs;
  if (typeof exported === 'number') return exported;
  return detector.causalDelayMs;
}

/**
 * `pre`: the activity and median grid step of this very stream, when the caller already has them (the live engine:
 * the bundle's single activity filter gives the same signal for both detectors, see the module docstring).
 */
export function restDetails(stream: Stream, detector: DetectorBundle, pre?: { readonly act: Float64Array; readonly dtMs: number }): RestDetails {
  const t = stream.t;
  const n = t.length;
  const p = detector.grip.restParams;
  const dt = n >= 2 ? (pre ? pre.dtMs : medianStep(t)) : NOMINAL_STEP_MS;
  const nLook = boutConstants(p, dt, restLagMs(detector)).nLook;
  if (n === 0) return { restUpZ: NaN, nSel: 0, fallback: false, firstOnsetMs: Infinity, nDefaultBouts: 0, firstBout: null, nLook };
  if (p.smooth_hz !== detector.segParams.smooth_hz || p.causal !== detector.segParams.causal) {
    throw new Error('detect: grip.restParams needs its own activity filter, which the bundle does not export');
  }
  const s = pre ? pre.act : activity(t, stream.gyro, p, { sos: detector.activitySos, zi: detector.activityZi });
  const segs = findBouts(t, s, p, restLagMs(detector), n >= 2 ? dt : undefined);
  const firstOn = segs.length ? segs[0].onsetMs : Infinity;
  const upz = new Float64Array(n);
  const ok = new Uint8Array(n);
  const f = stream.f.data;
  for (let i = 0; i < n; i++) {
    const nrm = norm3(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
    ok[i] = Number.isFinite(nrm) && nrm > 0 ? 1 : 0;
    upz[i] = ok[i] ? f[3 * i + 2] / nrm : NaN;
  }
  const fsHz = detector.constants.FS_HZ ?? 60.0;
  const minN = Math.max(1, roundHalfEven(detector.grip.restMinS * fsHz));
  let sel: number[] = [];
  for (let i = 0; i < n; i++) if (ok[i] && t[i] < firstOn && s[i] < p.lo_dps) sel.push(upz[i]);
  let fallback = false;
  if (sel.length < minN) {
    fallback = true;
    const until = t[0] + detector.grip.restMinS * 1000.0;
    sel = [];
    for (let i = 0; i < n; i++) if (ok[i] && t[i] < until) sel.push(upz[i]);
  }
  return {
    restUpZ: sel.length ? npMedian(sel) : NaN,
    nSel: sel.length,
    fallback,
    firstOnsetMs: firstOn,
    nDefaultBouts: segs.length,
    firstBout: segs.length ? segs[0] : null,
    nLook,
  };
}

/**
 * Live: rest_up_z of a prefix is final once the first DEFAULT bout is frozen (ended, look-ahead in the buffer, settled):
 * it only reads samples before that bout's onset, and neither the bout's onset nor its qualification can change.
 */
export function restIsFinal(r: RestDetails, stream: Stream, settleMs: number): boolean {
  const n = stream.t.length;
  return r.firstBout !== null && n > 0 && isFrozen(r.firstBout, r.nLook, n, stream.t[n - 1], settleMs);
}

/** m4_detect.rest_up_z with the bundle's grip parameters (NaN -> null). */
export function restUpZ(stream: Stream, detector: DetectorBundle): number | null {
  const r = restDetails(stream, detector).restUpZ;
  return Number.isNaN(r) ? null : r;
}

/** The gate of predict_stream: true when the rest pose is known and below grip.gateUpz (null gate = off). */
export function gripBlocked(rest: number | null, detector: DetectorBundle): boolean {
  const gate = detector.grip.gateUpz;
  return gate !== null && rest !== null && Number.isFinite(rest) && rest < gate;
}
