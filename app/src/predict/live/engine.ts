/**
 * THE SEAM between the UI and the engine: the only module the UI asks for an engine, the model list and the
 * production normalisation. Every screen reads EngineBinding, never detect/, models/ or pipeline/ directly.
 *
 * The real engine (docs/stage3/APP_SPEC.md 4.2):
 *   pipeline/  LiveBuffer (the recorder's rounding) and the incremental streamFrom (bit for bit stream_from_record)
 *   detect/    lane m4's cue-free detector, attempt rule, grip gate and stopping rule (makeCreateSessionEngine)
 *   models/    the four bundled models, each with its own calibrator and k-attempt aggregation
 *
 * The detector, the attempt threshold, the grip gate and the stopping thresholds (p_stop 0.9, k_max 5, p_answer
 * 0.5) are lane m4's: they were fitted for lane m4's own hand classifier and are applied unchanged to every model.
 * Lane prod, which would refit them for the final CNN (and for PHYS6 and the yaw reference), has not run.
 */
import { makeCreateSessionEngine, type EngineDeps } from '../detect/engine';
import { detector, loadModels, specs } from '../models/registry';
import { createIncrementalStreamFrom, pipeline, productionNormalization } from '../pipeline';
import type { BrowserEngine, CreateSessionEngine, HandModel, ModelSpec, StreamNormalization } from '../types';

export interface EngineBinding {
  /** 'real': detect/ + models/ + pipeline/ (the only kind left; the footer and data-engine show it) */
  readonly kind: 'real';
  /** registry order: cnn, m4, phys6, yaw */
  readonly specs: readonly ModelSpec[];
  readonly models: () => HandModel[];
  readonly createSessionEngine: CreateSessionEngine;
  /** the production normalisation of an engine (webkit gravity sign -1, every other engine +1; W3C deg/s) */
  readonly normalization: (engine: BrowserEngine | null) => StreamNormalization;
}

/**
 * What one engine needs: the pipeline with its own incremental streamFrom (about 0.4 ms per tick instead of a full
 * recomputation; same numbers bit for bit; it restarts by itself on reset) and lane m4's detector bundle.
 * Exported for the parity tests, which wrap these functions to read the intermediate values of the live path.
 */
export function engineDeps(): EngineDeps {
  return { pipeline: { ...pipeline, streamFrom: createIncrementalStreamFrom() }, detector: detector() };
}

/**
 * The page runs a new attempt's models (the CNN is most of the cost of a freeze) once the phone is still again, at
 * most this long after the freeze, so the work does not stall the sensor events of the next movement. Same numbers.
 */
export const DEFER_SCORING_MS = 1500;

/** one engine per stream (Start, Again, a new demo script): each gets its own incremental stream builder */
const createSessionEngine: CreateSessionEngine = (options) =>
  makeCreateSessionEngine(engineDeps())({ deferScoringMs: DEFER_SCORING_MS, ...options });

export const binding: EngineBinding = {
  kind: 'real',
  specs,
  models: loadModels,
  createSessionEngine,
  normalization: productionNormalization,
};
