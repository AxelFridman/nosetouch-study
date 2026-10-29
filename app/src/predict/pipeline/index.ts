/**
 * pipeline/: raw DeviceMotion samples -> the standardised 60 Hz stream -> canonical channels -> the (17, 64)
 * float32 approach window, a port of stage2/standardize.py and the motionlab preprocessing it calls
 * (docs/stage3/PORT_SPEC.md sections 1 and 2.3-2.7). Depends on nothing else in predict/ except the types.
 *
 * Live use: push samples into a LiveBuffer, then per tick either streamFrom(buffer.raw(), norm) (O(n): about 14 ms
 * at the 3-minute cap on a desktop) or its incremental drop-in createIncrementalStreamFrom() (about 0.4 ms), or
 * createStreamBuilder(norm).update(buffer.raw()) directly. All give the same numbers, bit for bit, as one streamFrom
 * call on the record the collector would have stored for the same samples. For the session engine:
 *   pipeline: { ...pipeline, streamFrom: createIncrementalStreamFrom() }   (one per engine / stream)
 *
 * Parity (pipeline tests, 5 committed synthetic + 40 local real fixtures): the stream, every crop, the gravity
 * counts and magnitude and all 171 (17, 64) float32 windows are bit for bit Python's; closure_deg differs by at most
 * 1.4e-14 deg (V8's atan2 / sin / cos against the C runtime's, one ulp now and then).
 */
import type { PipelineApi } from '../types';
import { createLiveBuffer } from './liveBuffer';
import { productionNormalization } from './normalize';
import { rawMotionFromRecord } from './record';
import { canonical, cropStream, streamFrom, timeNormalised } from './standardize';

export const pipeline: PipelineApi = {
  createLiveBuffer,
  rawMotionFromRecord,
  productionNormalization,
  streamFrom,
  cropStream,
  canonical,
  timeNormalised,
};

export { createLiveBuffer } from './liveBuffer';
export { ENGINE_GRAVITY_SIGN, productionNormalization, RR_MAPPINGS } from './normalize';
export { rawMotionFromRecord, rawPrefix } from './record';
export {
  canonical,
  cropStream,
  PipelineError,
  signedMotion,
  streamFrom,
  timeNormalised,
  windowFor,
  type CanonicalDetail,
  type CanonicalOptions,
} from './standardize';
export { createIncrementalStreamFrom, createStreamBuilder, type StreamBuilder, type StreamBuilderStats } from './incremental';
export { gyroGravity, integrateGyro, quaternionsToMatrices, type GyroGravity } from './gravity';
export { fillGapsLinear, finiteRuns, LOWPASS } from './preprocess';
export {
  angleDeg3,
  fma,
  interpBetween,
  linspace,
  makeSos,
  norm3,
  norm3Dot,
  npInterp,
  scaledZi,
  searchLeft,
  searchRight,
  sosStep,
  sosfiltfiltRun,
  type Sos,
} from './numeric';
export * from './constants';
