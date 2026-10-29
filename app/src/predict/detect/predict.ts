/**
 * m4_production.predict_stream (analysis/stage2/lanes/m4_production.py:81-119) for one COMPLETE stream, with any
 * HandModel as the hand classifier (lane m4's own model reproduces the Python exactly; the other models get lane
 * m4's detector, attempt rule, grip check and stopping thresholds, which were fitted for m4's classifier: lane
 * prod, which would refit them for the final CNN, has not run).
 *
 *   1. segment table: bouts -> window bounds -> canonical window -> features -> P(approach / return / other);
 *   2. attempts: the production attempt rule (no pairing); each attempt's window goes to the hand model;
 *   3. the grip check (rest_up_z of the stream with the DEFAULT detector, below the gate -> 'grip', P 0.5);
 *   4. the stopping rule over the attempts in decision order (offline: the stream has ended).
 */
import type {
  Decision,
  DecisionStatus,
  DetectorBundle,
  HandModel,
  ModelAttemptOutput,
  PipelineApi,
  RawMotion,
  Stream,
  StreamNormalization,
  StreamPrediction,
} from '../types';
import { activity } from './activity';
import type { WindowPipeline } from './canonical';
import { gripBlocked, restDetails, type RestDetails } from './grip';
import { segmentRows, type SegmentRowDetail } from './rows';
import { decisionOrder, gripDecision, isAvailable, kCurve, stopOnCurve } from './stopping';

/** Every intermediate of one complete stream (what predictStream summarises; the debug view and the tests). */
export interface StreamAnalysis {
  readonly activity: Float64Array;
  readonly rows: readonly SegmentRowDetail[];
  /** the attempts in decision order, with the model's output for each */
  readonly attempts: readonly { readonly row: SegmentRowDetail; readonly output: ModelAttemptOutput }[];
  readonly rest: RestDetails;
  readonly grip: boolean;
  /** P_1 .. P_K of the model over its available attempts */
  readonly kCurve: readonly number[];
  readonly decision: Decision;
}

/** predict_stream's steps on a standardised stream (streamEnded: true offline; false gives the live 'watching'). */
export function analyseStream(
  stream: Stream,
  model: HandModel,
  detector: DetectorBundle,
  pipeline: WindowPipeline,
  streamEnded = true,
): StreamAnalysis {
  const act = activity(stream.t, stream.gyro, detector.segParams, { sos: detector.activitySos, zi: detector.activityZi });
  const rows = segmentRows(stream, detector, pipeline, act);
  const attemptRows = decisionOrder(
    rows.filter((r) => r.isAttempt).map((row) => ({ decidedMs: row.bout.decidedMs, seg: row.bout.index, row })),
  );
  const attempts = attemptRows.map(({ row }) => {
    if (row.window === null) throw new Error('detect: an attempt without a window'); // isAttempt needs canonicalOk
    return { row, output: model.predict({ window: row.window }) };
  });
  const rest = restDetails(stream, detector);
  const grip = gripBlocked(Number.isNaN(rest.restUpZ) ? null : rest.restUpZ, detector);
  const avail = attempts.filter((a) => isAvailable(a.output));
  const pk = kCurve(model, avail.map((a) => a.output), detector.stopping.kMax);
  const decision = grip
    ? gripDecision(model.spec.id)
    : stopOnCurve(model.spec.id, pk, avail.map((a) => a.row.bout.decidedMs), detector.stopping, streamEnded);
  return { activity: act, rows, attempts, rest, grip, kCurve: pk, decision };
}

/** DetectApi.predictStream's core on a standardised stream (the parity tests start from the fixture's stream). */
export function predictFromStream(stream: Stream, model: HandModel, detector: DetectorBundle, pipeline: WindowPipeline): StreamPrediction {
  const a = analyseStream(stream, model, detector, pipeline, true);
  const d = a.decision;
  const status = d.status as Exclude<DecisionStatus, 'watching'>;
  return {
    modelId: model.spec.id,
    pRight: d.pRight,
    percentRight: d.percentRight,
    answered: d.answered,
    status,
    nAttemptsUsed: d.nUsed,
    decidedMs: d.decidedMs,
    restUpZ: Number.isNaN(a.rest.restUpZ) ? null : a.rest.restUpZ,
    gripWarning: a.grip,
    // predict_stream lists every attempt (sorted by decided_ms) with 1 / (1 + exp(-z)), no clip (m4_production.py:111-115)
    attempts: a.attempts.map(({ row, output }) => ({
      onsetMs: row.bout.onsetMs,
      endMs: row.bout.endMs,
      decidedMs: row.bout.decidedMs,
      pApproach: row.proba.approach,
      pRight: 1.0 / (1.0 + Math.exp(-output.z)),
    })),
    segments: a.rows.map((r) => ({
      onsetMs: r.bout.onsetMs,
      endMs: r.bout.endMs,
      decidedMs: r.bout.decidedMs,
      pApproach: r.proba.approach,
      pReturn: r.proba.return,
      pOther: r.proba.other,
    })),
  };
}

/** m4_production.predict_stream from the raw motion (standardised by the pipeline first). */
export function predictStream(
  raw: RawMotion,
  norm: StreamNormalization,
  model: HandModel,
  detector: DetectorBundle,
  pipeline: WindowPipeline & Pick<PipelineApi, 'streamFrom'>,
): StreamPrediction {
  return predictFromStream(pipeline.streamFrom(raw, norm), model, detector, pipeline);
}
