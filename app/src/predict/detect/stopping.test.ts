/**
 * The stopping rule (m4_online.stopping_rule + the page's per-model rule of analysis/stage3/reference.py `stopping`)
 * on hand-made P_k sequences, the attempt order, unavailable attempts and the grip decision.
 */
import { describe, expect, it } from 'vitest';
import type { AttemptResult, Bout, HandModel, ModelAttemptOutput, ModelId, StoppingParams } from '../types';
import { decisionOrder, gripDecision, handOf, isAvailable, percentRight, stopOnCurve, stoppingRule } from './stopping';

const PARAMS: StoppingParams = { pStop: 0.9, kMax: 5, pAnswer: 0.5 };

function bout(index: number, decidedMs: number): Bout {
  return {
    index,
    onsetMs: decidedMs - 900,
    endMs: decidedMs - 500,
    iOn: 0,
    iEnd: 0,
    decidedMs,
    stillBeforeS: 1,
    stillAfterS: 0.5,
    startedActive: false,
    ended: true,
    peakDps: 100,
  };
}

/** a model whose P_k is the mean of the attempts' p (enough to see which attempts were used) */
const meanModel = (id: ModelId): HandModel => ({
  spec: { id } as HandModel['spec'],
  predict: () => {
    throw new Error('unused');
  },
  aggregate: (o: readonly ModelAttemptOutput[]) => (o.length ? o.reduce((s, x) => s + x.pRight, 0) / o.length : 0.5),
});

function attempt(index: number, decidedMs: number, p: number, available = true): AttemptResult {
  const out = (modelId: ModelId): ModelAttemptOutput =>
    ({ modelId, zRaw: 0, pRight: p, z: available ? Math.log(p / (1 - p)) : NaN, available }) as ModelAttemptOutput;
  return {
    k: 0,
    bout: bout(index, decidedMs),
    pApproach: 0.8,
    pReturn: 0.1,
    pOther: 0.1,
    closureDeg: 2,
    restUpZBefore: 0.3,
    outputs: { cnn: out('cnn'), m4: out('m4'), phys6: out('phys6'), yaw: out('yaw') },
    frozenAtMs: decidedMs + 1000,
  };
}

describe('stopOnCurve', () => {
  it('answers as soon as max(P, 1 - P) >= p_stop', () => {
    const d = stopOnCurve('m4', [0.7, 0.95, 0.99], [100, 200, 300], PARAMS, true);
    expect([d.status, d.nUsed, d.pRight, d.answered, d.final, d.decidedMs, d.hand, d.percentRight]).toEqual([
      'confident', 2, 0.95, true, true, 200, 'right', 95,
    ]);
    expect(d.trajectory).toEqual([0.7, 0.95]);
    expect(stopOnCurve('m4', [0.05], [100], PARAMS, false).hand).toBe('left');
  });

  it('no attempt: no_attempt offline, watching live', () => {
    expect(stopOnCurve('cnn', [], [], PARAMS, true)).toMatchObject({ status: 'no_attempt', pRight: 0.5, final: true, answered: false, decidedMs: null });
    expect(stopOnCurve('cnn', [], [], PARAMS, false)).toMatchObject({ status: 'watching', final: false, answered: false });
  });

  it('after k_max attempts it always answers (p_answer 0.5), even at P = 0.5 exactly', () => {
    const d = stopOnCurve('m4', [0.6, 0.55, 0.52, 0.51, 0.5, 0.99], [1, 2, 3, 4, 5, 6], PARAMS, false);
    expect([d.status, d.nUsed, d.pRight, d.answered, d.final, d.decidedMs, d.hand]).toEqual(['k_max', 5, 0.5, true, true, 5, null]);
  });

  it('fewer than k_max undecided attempts: stream_end offline, watching (provisional P) live', () => {
    expect(stopOnCurve('m4', [0.6, 0.7], [1, 2], PARAMS, true)).toMatchObject({ status: 'stream_end', nUsed: 2, pRight: 0.7, answered: true, final: true });
    expect(stopOnCurve('m4', [0.6, 0.7], [1, 2], PARAMS, false)).toMatchObject({ status: 'watching', nUsed: 2, pRight: 0.7, answered: false, final: false, decidedMs: 2 });
  });

  it('p_answer above 0.5 abstains after k_max', () => {
    const d = stopOnCurve('m4', [0.6, 0.6, 0.6, 0.6, 0.6], [1, 2, 3, 4, 5], { ...PARAMS, pAnswer: 0.7 }, true);
    expect([d.status, d.answered, d.final]).toEqual(['k_max', false, true]);
  });

  it('percent is Python round-half-even; the hand of exactly 0.5 is none', () => {
    expect([percentRight(0.125), percentRight(0.135), percentRight(0.875), percentRight(0.5), percentRight(0.995)]).toEqual([12, 14, 88, 50, 100]);
    expect(handOf(0.5)).toBeNull();
  });
});

describe('stoppingRule over frozen attempts', () => {
  it('orders the attempts by decision time then bout index, whatever the input order', () => {
    const a = [attempt(3, 900, 0.8), attempt(1, 300, 0.6), attempt(2, 300, 0.7)];
    expect(decisionOrder(a.map((x) => ({ decidedMs: x.bout.decidedMs, seg: x.bout.index }))).map((x) => x.seg)).toEqual([1, 2, 3]);
    const d = stoppingRule(meanModel('cnn'), a, PARAMS, true);
    expect(d.trajectory).toEqual([0.6, (0.6 + 0.7) / 2, (0.6 + 0.7 + 0.8) / 3]);
    expect(d.status).toBe('stream_end');
  });

  it('skips the attempts a model cannot score (for that model only)', () => {
    const a = [attempt(0, 100, 0.95, false), attempt(1, 200, 0.6), attempt(2, 300, 0.97)];
    const d = stoppingRule(meanModel('phys6'), a, PARAMS, true);
    expect([d.nUsed, d.status, d.decidedMs]).toEqual([2, 'stream_end', 300]);
    expect(d.trajectory).toEqual([0.6, (0.6 + 0.97) / 2]);
    expect(isAvailable(a[0].outputs.phys6)).toBe(false);
    expect(isAvailable(undefined)).toBe(false);
  });

  it('grip: P 0.5, not answered, final', () => {
    expect(gripDecision('yaw')).toMatchObject({ modelId: 'yaw', status: 'grip', pRight: 0.5, percentRight: 50, answered: false, final: true, nUsed: 0 });
  });
});
