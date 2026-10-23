/**
 * Stage-3 hook: the trained left/right model will plug in here so the same app can run live
 * predictions for the PI. Stage 1 ships no model on purpose (see docs/08_STAGE3_PREDICTOR.md).
 */
import type { Hand, TrialRecord } from './types';

export interface Prediction {
  hand: Hand;
  p_right: number; // calibrated probability that the RIGHT hand was used
  model_id: string;
}

export interface Predictor {
  readonly id: string;
  /** Uses only sensor streams + segmentation of `trial` (never trial.condition). */
  predict(trial: TrialRecord): Prediction;
}
