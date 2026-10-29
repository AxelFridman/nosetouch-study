/**
 * detect/: lane m4's cue-free production detector, ported from analysis/stage2/lanes (m4_detect, m4_e2e, m4_online,
 * m4_production). See docs/stage3/PORT_SPEC.md sections 2 and 3.5.
 *
 *   activity.ts    causal activity |w| (batch + ActivityFilter fed sample by sample)
 *   bouts.ts       hysteresis bouts + look-ahead confirmation (findBouts batch, BoutTracker / CausalDetector
 *                  fed sample by sample, isFrozen = the live freeze rule)
 *   rows.ts        window bounds and the segment table (the window itself comes from a WindowPipeline)
 *   features.ts    the 35 'robust' segment features (incl. the ZUPT kinematics)
 *   classifier.ts  P(approach / return / other) and the attempt rule
 *   grip.ts        rest_up_z and the grip gate
 *   stopping.ts    the stopping rule over any model's P_k
 *   predict.ts     predict_stream on a complete stream
 *   live.ts        the live freeze rule per tick, same-bout matching, the screen-away hint
 *   engine.ts      the SessionEngine (makeCreateSessionEngine({pipeline, detector}) -> createSessionEngine)
 *   canonical.ts   detect's own port of window_for (crop, canonical/gravity_v1, time_normalised) - see its header
 *
 * The detector parameters, the segment classifier, the attempt threshold and the stopping thresholds are lane m4's:
 * they were fitted for lane m4's own hand classifier. Lane prod, which would refit them for the final CNN (and the
 * other models), has not run.
 */
import type { DetectApi, DetectorBundle, PipelineApi, SegParams, Stream } from '../types';
import { activity } from './activity';
import { findBouts } from './bouts';
import type { WindowPipeline } from './canonical';
import { restUpZ } from './grip';
import { predictStream } from './predict';
import { segmentRows } from './rows';
import { stoppingRule } from './stopping';

export { ActivityFilter, MIN_ACTIVITY_N, activity, type ActivityFilterCoefs, type ActivitySample } from './activity';
export {
  BoutTracker,
  CausalDetector,
  NOMINAL_STEP_MS,
  boutConstants,
  findBouts,
  isFrozen,
  medianStep,
  type BoutConstants,
} from './bouts';
export { CHANNELS, canonical, cropStream, detectWindowPipeline, timeNormalised, type WindowPipeline } from './canonical';
export { ARGMAX_ORDER, argmaxClass, ignoredReason, isAttempt, segmentProba } from './classifier';
export { ALL_FEATURES, INSTRUCTED_STILLNESS_S, segmentFeatures } from './features';
export { gripBlocked, restDetails, restIsFinal, restUpZ, type RestDetails } from './grip';
export { DEFAULT_MAX_STREAM_MS, DEFAULT_SETTLE_MS, DEFAULT_TICK_MS, makeCreateSessionEngine, type EngineDeps } from './engine';
export { HINT_WINDOW_MS, SETTLE_MS, liveBouts, sameBout, screenAwayHint, type LiveBouts } from './live';
export { analyseStream, predictFromStream, predictStream, type StreamAnalysis } from './predict';
export { segmentRow, segmentRows, windowBounds, type SegmentRowDetail } from './rows';
export {
  decisionOrder,
  gripDecision,
  handOf,
  isAvailable,
  kCurve,
  percentRight,
  stopOnCurve,
  stoppingRule,
} from './stopping';

/** DetectApi without the live engine (createSessionEngine is wired by the integrator). */
export type DetectFunctions = Omit<DetectApi, 'createSessionEngine'>;

/**
 * detect/'s pure functions as DetectApi wants them, bound to the pipeline (window cutting, raw -> stream) and to
 * the exported detector bundle (the activity filter and its causal delay, which the DetectApi signatures do not
 * carry; both depend only on smooth_hz, the same 5 Hz for the fitted detector and the grip's DEFAULT one).
 */
export function createDetectFunctions(
  pipeline: WindowPipeline & Pick<PipelineApi, 'streamFrom'>,
  detector: DetectorBundle,
): DetectFunctions {
  const coefs = { sos: detector.activitySos, zi: detector.activityZi };
  const check = (p: SegParams): void => {
    if (p.smooth_hz !== detector.segParams.smooth_hz || p.causal !== detector.segParams.causal) {
      throw new Error(`detect: no exported activity filter for smooth_hz ${p.smooth_hz} (causal ${p.causal})`);
    }
  };
  return {
    activity: (stream: Stream, p: SegParams) => {
      check(p);
      return activity(stream.t, stream.gyro, p, coefs);
    },
    findBouts: (stream: Stream, p: SegParams, act?: Float64Array) => {
      check(p);
      return findBouts(stream.t, act ?? activity(stream.t, stream.gyro, p, coefs), p, detector.causalDelayMs);
    },
    segmentRows: (stream: Stream, det: DetectorBundle) => segmentRows(stream, det, pipeline),
    restUpZ: (stream: Stream, det: DetectorBundle) => restUpZ(stream, det),
    stoppingRule,
    predictStream: (raw, norm, model, det) => predictStream(raw, norm, model, det, pipeline),
  };
}
