/**
 * k-attempt aggregation: P_k, the P(right) after the first k attempts (decision order), each model with its own
 * rule, as analysis/stage3/runtime.py ModelSet.aggregate applies them:
 *
 *   sum_over_tau   (final CNN)  final_infer.aggregate (final_infer.py:177-184):
 *                  P_k = sigmoid(sum of logit(p_i, eps 1e-12) / tau_k), tau_1 = 1, k above the table uses the largest
 *                  key; no clip in this sigmoid.
 *   scaled_mean    (lane m4)    m4_online.aggregate 'scaled' (m4_online.py:38-48) with AggregateScale.scale
 *                  (m4_models.py:478-483): P_k = m4 sigmoid(s_k * mean z_i), argument clipped to +-40; s null = 1.
 *   mean / sum     (m4_online 'mean' / 'sum'): m4 sigmoid of the mean / the sum of z_i (no exported model uses them).
 *   sum_over_k_pow (PHYS6)      lane m1 fitted_alpha (runtime.Phys6.aggregate, m1_analysis.k_aggregation):
 *                  P_k = sigmoid(sum z_i / k^alpha), no clip.
 *   mean_p         (yaw ref.)   the harness k-curve (runtime.YawRef.aggregate, cv.k_attempt_curve): P_k = mean p_i.
 *                  Never fitted, so P_k is NOT calibrated for k >= 2.
 *
 * An attempt the model cannot score (available false, or a non-finite p or z) is skipped for that model only
 * (runtime.ModelSet.aggregate filters o['available']); with no usable attempt P = 0.5.
 */
import type { AggregationSpec, ModelAttemptOutput } from '../types';
import { logitLog1p, pairwiseSum, sigmoid, sigmoidClipped } from './numeric';

/** m4_models.sigmoid's clip (m4_models.py:43) */
export const M4_SIGMOID_CLIP = 40;
/** final_infer.logit's eps (final_infer.py:36) */
export const FINAL_LOGIT_EPS = 1e-12;

/** the attempt counts for this model */
export function isUsable(o: ModelAttemptOutput): boolean {
  return o.available !== false && Number.isFinite(o.pRight) && Number.isFinite(o.z);
}

/** a fitted table keyed by '1', '2', ... -> the value at k, or at the largest key when k is beyond it */
export function tableAt(table: Readonly<Record<string, number>>, k: number): number | undefined {
  const keys = Object.keys(table)
    .map(Number)
    .filter((n) => Number.isInteger(n));
  if (keys.length === 0) return undefined;
  const v = table[String(k)];
  if (v !== undefined) return v;
  return table[String(Math.max(...keys))];
}

/** P_k of the given outputs (k = the number of usable ones) under one aggregation rule. */
export function aggregateOutputs(spec: AggregationSpec, outputs: readonly ModelAttemptOutput[]): number {
  const used = outputs.filter(isUsable);
  const k = used.length;
  if (k === 0) return 0.5;
  switch (spec.kind) {
    case 'sum_over_tau': {
      // final_infer.aggregate recomputes the logits from the calibrated p (eps 1e-12) and sums them (np.sum)
      const z = used.map((o) => logitLog1p(o.pRight, FINAL_LOGIT_EPS));
      const tau = k > 1 ? tableAt(spec.tau, k) ?? 1.0 : 1.0;
      return sigmoid(pairwiseSum(z, 0, k) / tau);
    }
    case 'scaled_mean': {
      // AggregateScale.scale(k): s[k], or s[max k] beyond the table; an empty table or no scale (agg None) is 1
      const s = spec.s ? tableAt(spec.s, k) ?? 1.0 : 1.0;
      const z = used.map((o) => o.z);
      return sigmoidClipped(s * (pairwiseSum(z, 0, k) / k), M4_SIGMOID_CLIP);
    }
    case 'mean': {
      const z = used.map((o) => o.z);
      return sigmoidClipped(pairwiseSum(z, 0, k) / k, M4_SIGMOID_CLIP);
    }
    case 'sum': {
      const z = used.map((o) => o.z);
      return sigmoidClipped(pairwiseSum(z, 0, k), M4_SIGMOID_CLIP);
    }
    case 'sum_over_k_pow': {
      const z = used.map((o) => o.z);
      return sigmoid(pairwiseSum(z, 0, k) / Math.pow(k, spec.alpha));
    }
    case 'mean_p': {
      const p = used.map((o) => o.pRight);
      return pairwiseSum(p, 0, k) / k;
    }
    default: {
      const unknown: never = spec;
      throw new RangeError(`unknown aggregation ${JSON.stringify((unknown as { kind?: unknown }).kind)}`);
    }
  }
}

/** P_1 .. P_K of the usable outputs, in order (the k-curve the fixtures store). */
export function kCurve(spec: AggregationSpec, outputs: readonly ModelAttemptOutput[]): number[] {
  const used = outputs.filter(isUsable);
  const out: number[] = [];
  for (let k = 1; k <= used.length; k++) out.push(aggregateOutputs(spec, used.slice(0, k)));
  return out;
}
