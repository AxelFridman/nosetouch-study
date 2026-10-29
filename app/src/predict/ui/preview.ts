/**
 * DEV-ONLY fixed states of every screen (?predict=1&preview=<name>), for screenshot reviews and layout
 * tests at phone sizes. mount.tsx imports this module only under import.meta.env.DEV, so production
 * builds do not contain it. The numbers are made up; they never come from a recording.
 */
import type { PreviewState } from '../PredictApp';
import type {
  AttemptResult,
  Bout,
  Decision,
  DecisionStatus,
  EngineSnapshot,
  HandModel,
  ModelAttemptOutput,
  ModelComparison,
  ModelId,
  SourceFailure,
  StoppingParams,
} from '../types';
import { percentOf } from './format';

const STOPPING: StoppingParams = { pStop: 0.9, kMax: 5, pAnswer: 0.5 };
const logit = (p: number): number => Math.log(p / (1 - p));

function bout(i: number): Bout {
  const onsetMs = 2500 + i * 6000;
  return {
    index: 2 * i,
    onsetMs,
    endMs: onsetMs + 900,
    iOn: Math.round(onsetMs / 16.67),
    iEnd: Math.round((onsetMs + 900) / 16.67),
    decidedMs: onsetMs + 1400,
    stillBeforeS: 1.5,
    stillAfterS: 0.5,
    startedActive: false,
    ended: true,
    peakDps: 180,
  };
}

/** attempts whose per-model P(right) are `ps[k][id]` */
function attemptsOf(ps: readonly Readonly<Record<ModelId, number>>[]): AttemptResult[] {
  return ps.map((row, i) => {
    const outputs = {} as Record<ModelId, ModelAttemptOutput>;
    for (const id of Object.keys(row) as ModelId[]) outputs[id] = { modelId: id, zRaw: logit(row[id]), pRight: row[id], z: logit(row[id]) };
    return {
      k: i + 1,
      bout: bout(i),
      pApproach: 0.93,
      pReturn: 0.04,
      pOther: 0.03,
      closureDeg: 4.2,
      restUpZBefore: 0.31,
      outputs,
      frozenAtMs: bout(i).decidedMs + 1000,
    };
  });
}

function snapshotOf(
  models: readonly HandModel[],
  driver: ModelId,
  ps: readonly Readonly<Record<ModelId, number>>[],
  phase: EngineSnapshot['phase'],
  status: DecisionStatus,
  answered: boolean,
  extra: Partial<EngineSnapshot> = {},
): EngineSnapshot {
  const attempts = attemptsOf(ps);
  const model = models.find((m) => m.spec.id === driver) ?? models[0];
  const outs = attempts.map((a) => a.outputs[model.spec.id]);
  const p = status === 'grip' ? 0.5 : model.aggregate(outs);
  const trajectory = outs.map((_, k) => model.aggregate(outs.slice(0, k + 1)));
  const decision: Decision = {
    modelId: model.spec.id,
    pRight: p,
    percentRight: percentOf(p),
    nUsed: status === 'grip' ? 0 : attempts.length,
    status,
    final: phase === 'final' || phase === 'grip',
    answered,
    hand: p > 0.5 ? 'right' : p < 0.5 ? 'left' : null,
    decidedMs: attempts.length ? attempts[attempts.length - 1].bout.decidedMs : null,
    trajectory,
  };
  const compare: ModelComparison[] = models.map((m) => {
    const o = attempts.map((a) => a.outputs[m.spec.id]);
    const pm = m.aggregate(o);
    return { modelId: m.spec.id, pRight: pm, percentRight: percentOf(pm), k: o.length, confident: Math.max(pm, 1 - pm) >= STOPPING.pStop };
  });
  return {
    phase,
    tMs: 14_200,
    nSamples: 852,
    rateHz: 60,
    still: true,
    screenAwayHint: false,
    restUpZ: 0.31,
    attempts,
    ignored: [],
    driver: model.spec.id,
    decision,
    compare,
    stopping: STOPPING,
    lastEvent: null,
    ...extra,
  };
}

const RIGHT_1: Record<ModelId, number> = { cnn: 0.81, m4: 0.74, phys6: 0.69, yaw: 0.66 };
const RIGHT_2: Record<ModelId, number> = { cnn: 0.86, m4: 0.83, phys6: 0.71, yaw: 0.7 };
const LEFT_1: Record<ModelId, number> = { cnn: 0.03, m4: 0.08, phys6: 0.12, yaw: 0.2 };
const FLAT: Record<ModelId, number> = { cnn: 0.5, m4: 0.48, phys6: 0.52, yaw: 0.5 };

export const PREVIEW_NAMES = [
  'start',
  'permission',
  'watching',
  'moving',
  'hint',
  'provisional',
  'answer',
  'answer-left',
  'abstain',
  'abstain-flat',
  'grip',
  'unsupported',
  'denied',
  'inapp',
] as const;

export function previewState(name: string, models: readonly HandModel[], driver: ModelId): PreviewState | null {
  const running = (snapshot: EngineSnapshot): PreviewState => ({ stage: 'running', failure: null, snapshot });
  const failed = (failure: SourceFailure): PreviewState => ({ stage: 'start', failure, snapshot: null, preflight: null });
  switch (name) {
    case 'start':
      return { stage: 'start', failure: null, snapshot: null };
    case 'permission':
      return { stage: 'permission', failure: null, snapshot: null };
    case 'watching':
      return running(snapshotOf(models, driver, [], 'watching', 'watching', false, { tMs: 1800, nSamples: 108 }));
    case 'moving':
      return running(snapshotOf(models, driver, [], 'watching', 'watching', false, { still: false }));
    case 'hint':
      return running(snapshotOf(models, driver, [], 'watching', 'watching', false, { screenAwayHint: true }));
    case 'provisional':
      return running(snapshotOf(models, driver, [RIGHT_1], 'watching', 'watching', false));
    case 'answer':
      return running(snapshotOf(models, driver, [RIGHT_1, RIGHT_2], 'final', 'confident', true));
    case 'answer-left':
      return running(snapshotOf(models, driver, [LEFT_1], 'final', 'confident', true));
    case 'abstain':
      // what the rule abstains on in practice: no attempt before the time limit
      return running(snapshotOf(models, driver, [], 'final', 'no_attempt', false));
    case 'abstain-flat':
      return running(snapshotOf(models, driver, [FLAT], 'final', 'stream_end', false));
    case 'grip':
      return running(snapshotOf(models, driver, [], 'grip', 'grip', false, { restUpZ: -0.42 }));
    case 'unsupported':
      return failed('no_events');
    case 'denied':
      return failed('permission_denied');
    case 'inapp':
      return failed('in_app_browser');
    default:
      return null;
  }
}
