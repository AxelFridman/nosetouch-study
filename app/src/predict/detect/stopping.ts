/**
 * The sequential stopping rule over attempts (m4_online.stopping_rule, analysis/stage2/lanes/m4_online.py:51-66),
 * applied to ANY model's own P_k (the page's rule, analysis/stage3/reference.py `stopping`; PORT_SPEC 3.5):
 *
 *   attempts in decision order (decided_ms, then the bout index); the attempts a model cannot score are skipped
 *   for that model only;
 *   for k = 1 .. min(k_max, K):  P_k = the model's aggregation of its first k outputs;
 *                                answer at once when max(P_k, 1 - P_k) >= p_stop        -> 'confident'
 *   no attempt                                                                          -> 'no_attempt' (P 0.5)
 *   live and k < k_max                                                                  -> 'watching' (provisional)
 *   otherwise answered when max(P, 1 - P) >= p_answer (0.5: always)                     -> 'k_max' | 'stream_end'
 *
 * The thresholds (p_stop 0.9, k_max 5, p_answer 0.5) come from the exported m4 bundle: they were chosen for lane m4's
 * own classifier and are applied unchanged to every model (lane prod, which would refit them for the final CNN, has
 * not run).
 */
import type { AttemptResult, Decision, DecisionStatus, Hand, HandModel, ModelAttemptOutput, ModelId, StoppingParams } from '../types';
import { roundHalfEven } from './numeric';

/** int(round(100 * p)) with Python's round-half-even (m4_production.py:117). */
export function percentRight(p: number): number {
  return roundHalfEven(100 * p);
}

/** The more likely hand: P > 0.5 right, < 0.5 left, exactly 0.5 (or NaN) none. */
export function handOf(p: number): Hand | null {
  return p > 0.5 ? 'right' : p < 0.5 ? 'left' : null;
}

/**
 * Whether a model could score an attempt: its output exists, is not flagged unavailable (PHYS6 on a horizontal
 * screen, a NaN window for the CNN or the yaw reference) and has a finite calibrated logit.
 */
export function isAvailable(o: ModelAttemptOutput | undefined): boolean {
  if (o === undefined) return false;
  return o.available !== false && Number.isFinite(o.z);
}

/** One attempt as the stopping rule sees it. */
export interface StopInput {
  readonly decidedMs: number;
  /** the bout index (the tie-break of the decision order) */
  readonly seg: number;
}

/** The decision order of m4_online.stopping_rule: by decided_ms, ties by the bout index (stable). */
export function decisionOrder<T extends StopInput>(attempts: readonly T[]): T[] {
  return attempts.slice().sort((a, b) => a.decidedMs - b.decidedMs || a.seg - b.seg);
}

/**
 * The rule on precomputed P_1 .. P_K of one model (reference.py `stopping`); `decided[k - 1]` is the decision time
 * of the k-th used attempt. With streamEnded false an undecided stream is 'watching' (the page's live view).
 */
export function stopOnCurve(
  modelId: ModelId,
  pk: readonly number[],
  decided: readonly number[],
  params: StoppingParams,
  streamEnded: boolean,
): Decision {
  const make = (p: number, k: number, status: DecisionStatus, answered: boolean, final: boolean, decidedMs: number | null): Decision => ({
    modelId,
    pRight: p,
    percentRight: percentRight(p),
    nUsed: k,
    status,
    final,
    answered,
    hand: handOf(p),
    decidedMs,
    trajectory: pk.slice(0, k),
  });
  let p = 0.5;
  let k = 0;
  const kTop = Math.min(params.kMax, pk.length);
  for (k = 1; k <= kTop; k++) {
    p = pk[k - 1];
    if (Math.max(p, 1 - p) >= params.pStop) return make(p, k, 'confident', true, true, decided[k - 1]);
  }
  k = kTop; // Python's loop variable keeps its last value (0 when the loop did not run)
  if (k === 0) return make(0.5, 0, streamEnded ? 'no_attempt' : 'watching', false, streamEnded, null);
  if (k < params.kMax && !streamEnded) return make(p, k, 'watching', false, false, decided[k - 1]);
  const answered = Math.max(p, 1 - p) >= params.pAnswer;
  return make(p, k, k >= params.kMax ? 'k_max' : 'stream_end', answered, true, decided[k - 1]);
}

/** P_1 .. P_K of a model over its available outputs (the model's own k-aggregation; k = 0 is not included). */
export function kCurve(model: HandModel, outputs: readonly ModelAttemptOutput[], kMax = Infinity): number[] {
  const pk: number[] = [];
  const top = Math.min(kMax, outputs.length);
  for (let k = 1; k <= top; k++) pk.push(model.aggregate(outputs.slice(0, k)));
  return pk;
}

/**
 * DetectApi.stoppingRule: the model's decision over the engine's frozen attempts (any order; they are put in
 * decision order here). Only the first k_max available attempts are aggregated.
 */
export function stoppingRule(
  model: HandModel,
  attempts: readonly AttemptResult[],
  params: StoppingParams,
  streamEnded: boolean,
): Decision {
  const id = model.spec.id;
  const ordered = decisionOrder(attempts.map((a) => ({ decidedMs: a.bout.decidedMs, seg: a.bout.index, a })));
  const avail = ordered.filter((x) => isAvailable(x.a.outputs[id]));
  const outs = avail.map((x) => x.a.outputs[id]);
  const pk = kCurve(model, outs, params.kMax);
  return stopOnCurve(id, pk, avail.map((x) => x.decidedMs), params, streamEnded);
}

/** The decision under a failed grip check (m4_production.py:106-107): P 0.5, not answered, final. */
export function gripDecision(modelId: ModelId): Decision {
  return {
    modelId,
    pRight: 0.5,
    percentRight: 50,
    nUsed: 0,
    status: 'grip',
    final: true,
    answered: false,
    hand: null,
    decidedMs: null,
    trajectory: [],
  };
}
