/**
 * One row per bout: its window bounds, canonical window, features, P(class) and the attempt rule
 * (m4_detect.segment_table + window_for, analysis/stage2/lanes/m4_detect.py:242-256, 327-350; then
 * m4_e2e.FittedPipeline.segment_proba / attempts).
 *
 * The window itself is cut by the pipeline (WindowPipeline: cropStream, canonical, timeNormalised); detect/ decides
 * the bounds: the crop is [onset - crop_s, end + lookahead_s] so that the window uses no sample the causal detector
 * has not seen when it confirms the bout (the crop ends about 456 ms after the bout end, before decided_ms).
 */
import type { Bout, Canonical, DetectorBundle, SegParams, SegmentRow, Stream } from '../types';
import { activity } from './activity';
import { findBouts } from './bouts';
import type { WindowPipeline } from './canonical';
import { argmaxClass, isAttempt, segmentProba } from './classifier';
import { ALL_FEATURES, segmentFeatures } from './features';

/** window_for's crop bounds of a bout (m4_detect.py:253-254; exact expressions, they decide float64 ties). */
export function windowBounds(onsetMs: number, endMs: number, p: SegParams): { startMs: number; stopMs: number } {
  return { startMs: onsetMs - p.crop_s * 1000.0, stopMs: endMs + p.lookahead_s * 1000.0 };
}

/** A SegmentRow plus the Canonical it was computed from (null when canonical() refused the bounds). */
export interface SegmentRowDetail extends SegmentRow {
  readonly canonical: Canonical | null;
  /** the crop bounds handed to the pipeline */
  readonly crop: { readonly startMs: number; readonly stopMs: number };
}

/** Every feature name as NaN (segment_table's row of a bout whose canonical() failed: m4_detect.py:340, 346-348). */
function nanFeatures(detector: DetectorBundle): Record<string, number> {
  return Object.fromEntries([...ALL_FEATURES, ...detector.segmentClassifier.features].map((k) => [k, NaN]));
}

/**
 * segment_table's row of one bout, classified (the try/except ValueError of m4_detect.py:335-340). `prev` is the
 * previous bout of the stream (segment_table passes it whether or not its own canonical succeeded).
 */
export function segmentRow(
  stream: Stream,
  bout: Bout,
  detector: DetectorBundle,
  pipeline: WindowPipeline,
  prev: Bout | null = null,
): SegmentRowDetail {
  const p = detector.segParams;
  const crop = windowBounds(bout.onsetMs, bout.endMs, p);
  let can: Canonical | null = null;
  let window: SegmentRow['window'] = null;
  let features: Record<string, number>;
  try {
    can = pipeline.canonical(pipeline.cropStream(stream, crop.startMs, crop.stopMs), bout.onsetMs, bout.endMs);
    window = pipeline.timeNormalised(can, bout.onsetMs, bout.endMs);
    features = segmentFeatures(can, bout, prev);
  } catch (e) {
    if (!(e instanceof RangeError)) throw e; // pipeline: RangeError where Python raises ValueError
    can = null;
    window = null;
    features = nanFeatures(detector);
  }
  const canonicalOk = can !== null;
  const proba = segmentProba(detector.segmentClassifier, features);
  const cls = argmaxClass(proba);
  return {
    bout,
    canonicalOk,
    features,
    window,
    closureDeg: can !== null ? can.closureDeg : NaN,
    proba,
    cls,
    isAttempt: isAttempt(cls, proba, canonicalOk, detector.attemptRule),
    canonical: can,
    crop,
  };
}

/** m4_detect.segment_table + the classifier + the attempt rule over a complete stream (time order). */
export function segmentRows(stream: Stream, detector: DetectorBundle, pipeline: WindowPipeline, act?: Float64Array): SegmentRowDetail[] {
  const p = detector.segParams;
  const a = act ?? activity(stream.t, stream.gyro, p, { sos: detector.activitySos, zi: detector.activityZi });
  const bouts = findBouts(stream.t, a, p, detector.causalDelayMs);
  return bouts.map((b, k) => segmentRow(stream, b, detector, pipeline, k > 0 ? bouts[k - 1] : null));
}
